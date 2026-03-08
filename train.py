# train.py

import os
import csv
import argparse
import random
from typing import Any, Dict, List, Tuple, Union
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from modules.diffusion import ConditionalDiffusionModel
from src.utils import get_loader, compute_metrics


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_regression_dataset(name: str) -> bool:
    return name.lower() in {"mosi", "mosei", "sims"}


def move_to_device_list(xs: List[Any], device: torch.device) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for x in xs:
        if isinstance(x, torch.Tensor):
            out.append(x.to(device, non_blocking=True))
    return out


def corrupt_labels(y: torch.Tensor, task: "TaskSpec", label_noise: float, device: torch.device) -> torch.Tensor:
    """Inject label noise to intentionally weaken training."""
    if label_noise <= 0:
        return y
    y_noisy = y.clone().to(device)
    if task.task_type == "regression":
        return y_noisy.float() + torch.randn_like(y_noisy.float()) * label_noise

    if task.task_type == "binary":
        if y_noisy.dim() == 2 and y_noisy.size(1) == 2 and _is_one_hot(y_noisy):
            idx = y_noisy[:, 1].clone()
        else:
            idx = torch.clamp(y_noisy.view(-1), 0, 1)
        mask = (torch.rand_like(idx.float()) < label_noise)
        idx[mask] = 1 - idx[mask]
        if y_noisy.dim() == 2 and y_noisy.size(1) == 2:
            y_out = torch.zeros_like(y_noisy)
            y_out[:, 0] = 1 - idx
            y_out[:, 1] = idx
            return y_out
        return idx.view_as(y_noisy)

    # multiclass
    num_classes = task.num_classes
    idx = to_class_index(y_noisy, num_classes)
    mask = (torch.rand_like(idx.float()) < label_noise)
    rand_cls = torch.randint(0, num_classes, idx.shape, device=device)
    idx[mask] = rand_cls[mask]
    if y_noisy.dim() == 2 and y_noisy.size(1) == num_classes:
        return F.one_hot(idx, num_classes=num_classes).float()
    return idx.view_as(idx)


def maybe_shuffle_labels(y: torch.Tensor) -> torch.Tensor:
    if y.numel() == 0:
        return y
    flat = y.view(-1)
    perm = torch.randperm(flat.numel(), device=flat.device)
    return flat[perm].view_as(y)


def extract_batch(batch: Tuple[Any, ...]) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    if len(batch) == 3 and isinstance(batch[0], (list, tuple)):
        x_list, y, missing_mode = batch
        return list(x_list), y, missing_mode
    if len(batch) >= 5:
        text, audio, visual, y, missing_mode = batch[:5]
        x_list = [text, audio, visual]
        return x_list, y, missing_mode
    tensors = [b for b in batch if isinstance(b, torch.Tensor)]
    if len(tensors) < 5:
        raise ValueError(f"Unexpected batch format, got {len(batch)} elements.")
    x_list = tensors[:3]
    y = tensors[-2]
    missing_mode = tensors[-1]
    return x_list, y, missing_mode


def mask_to_index(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 2 and mask.size(-1) >= 3:
        return mask.argmax(dim=1)
    return mask.view(-1).long()


class TaskSpec:
    def __init__(self, task_type: str, num_classes: int = 1, class_weights: Union[None, torch.Tensor] = None):
        self.task_type = task_type
        self.num_classes = num_classes
        self.class_weights = class_weights

    def __repr__(self):
        if self.task_type == "multiclass":
            return f"TaskSpec(multiclass, num_classes={self.num_classes})"
        return f"TaskSpec({self.task_type})"


def _is_one_hot(mat: torch.Tensor) -> bool:
    if mat.dtype not in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        return False
    if mat.dim() != 2:
        return False
    if not torch.all((mat == 0) | (mat == 1)):
        return False
    row_sum = mat.sum(dim=1)
    return torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4)


def to_class_index(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    if y.dim() == 3 and y.size(2) == 2:
        y = y[:, :, 1]
    if y.dim() == 2 and y.size(1) == num_classes:
        return torch.argmax(y, dim=1).long()
    return y.view(-1).long()


@torch.no_grad()
def _estimate_class_weights(train_loader, device: torch.device, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    eps = 1e-6
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for batch in train_loader:
        _, y, _ = extract_batch(batch)
        idx = to_class_index(y, num_classes=num_classes).cpu()
        for c in range(num_classes):
            counts[c] += (idx == c).sum().item()
    freq = counts / counts.sum().clamp_min(1.0)
    weights = 1.0 / (freq + eps)
    weights = weights * (num_classes / weights.sum())
    weights = weights.to(device=device, dtype=torch.float32)
    return weights, counts.to(dtype=torch.float32)


def infer_task_spec(dataset_name: str, train_loader, device: torch.device) -> Tuple[TaskSpec, torch.Tensor]:
    if is_regression_dataset(dataset_name):
        return TaskSpec("regression", 1, None), None
    it = iter(train_loader)
    try:
        sample = next(it)
    except StopIteration:
        return TaskSpec("regression", 1, None), None
    _, y, _ = extract_batch(sample)
    y = y.to(device)

    if y.dim() == 3 and y.size(2) == 2:
        num_classes = y.size(1)
        class_weights, counts = _estimate_class_weights(train_loader, device, num_classes)
        print(f"[INFO] Class weights (multiclass): {class_weights.detach().cpu().tolist()}")
        return TaskSpec("multiclass", num_classes, class_weights), counts
    if y.dim() == 2 and y.size(1) > 1:
        num_classes = y.size(1)
        class_weights, counts = _estimate_class_weights(train_loader, device, num_classes)
        print(f"[INFO] Class weights (multiclass): {class_weights.detach().cpu().tolist()}")
        return TaskSpec("multiclass", num_classes, class_weights), counts

    y_flat = y.view(-1)
    num = y_flat.numel()
    if num > 0:
        zeros = (y_flat == 0).sum().item()
        ones = (y_flat == 1).sum().item()
        ratio = (zeros + ones) / float(num)
        if ratio > 0.95:
            return TaskSpec("binary", 2, None), None
    return TaskSpec("regression", 1, None), None


class AttentivePool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))

    def forward(self, x):
        w = self.proj(x).squeeze(-1)
        w = torch.softmax(w, dim=1)
        return (x * w.unsqueeze(-1)).sum(dim=1)


class TemporalBlock(nn.Module):
    def __init__(self, in_dim: int, encoder: str = "bilstm", hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        if encoder == "bilstm":
            self.rnn = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=1,
                                batch_first=True, bidirectional=True)
            self.attn = AttentivePool(hidden * 2)
            self.out_dim = hidden * 2
            self.do = nn.Dropout(dropout)
        else:
            self.rnn = None
            self.attn = None
            self.out_dim = in_dim

    def forward(self, x):
        if x.dim() == 2:
            return x
        if self.encoder == "bilstm":
            if hasattr(self.rnn, "flatten_parameters"):
                self.rnn.flatten_parameters()
            h, _ = self.rnn(x)
            h = self.do(h)
            return self.attn(h)
        return x.mean(dim=1)


class SingleHeadCrossAttn(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.do = nn.Dropout(dropout)

    def forward(self, q_vec: torch.Tensor, kv_list: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.q(q_vec)
        K = torch.stack([self.k(x) for x in kv_list], dim=1)
        V = torch.stack([self.v(x) for x in kv_list], dim=1)
        attn_logits = torch.einsum("bd,bnd->bn", q, K) * self.scale
        w = torch.softmax(attn_logits, dim=1)
        ctx = torch.einsum("bn,bnd->bd", w, V)
        return self.do(ctx), w


class CrossModalFusionHead(nn.Module):
    def __init__(
        self,
        in_dims: Tuple[int, int, int],
        hidden_dim: int = 256,
        out_dim: int = 1,
        dropout: float = 0.25,
        temporal_encoder: str = "none",
        temporal_hidden: int = 128,
        use_cross_attn: bool = False,
        cross_attn_mode: str = "t2av",
        enable_after: int = -1,
        tau0: float = 1.2,
        use_gate: bool = True,
        use_prompt: bool = True,
        prompt_dropout: float = 0.05,
    ):
        super().__init__()
        t_dim, a_dim, v_dim = in_dims

        self.t_temp = TemporalBlock(t_dim, temporal_encoder, temporal_hidden, dropout)
        self.a_temp = TemporalBlock(a_dim, temporal_encoder, temporal_hidden, dropout)
        self.v_temp = TemporalBlock(v_dim, temporal_encoder, temporal_hidden, dropout)

        self.t_proj = nn.Sequential(nn.LayerNorm(self.t_temp.out_dim), nn.Linear(self.t_temp.out_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.a_proj = nn.Sequential(nn.LayerNorm(self.a_temp.out_dim), nn.Linear(self.a_temp.out_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.v_proj = nn.Sequential(nn.LayerNorm(self.v_temp.out_dim), nn.Linear(self.v_temp.out_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))

        self.fuse_pre = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 6), nn.Sigmoid())
        self.use_gate = use_gate

        self.use_cross_attn = use_cross_attn
        self.cross_attn_mode = cross_attn_mode
        if use_cross_attn:
            self.attn_t = SingleHeadCrossAttn(hidden_dim, dropout)
            if cross_attn_mode == "all":
                self.attn_a = SingleHeadCrossAttn(hidden_dim, dropout)
                self.attn_v = SingleHeadCrossAttn(hidden_dim, dropout)
            self.ctx_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
            self.ctx_gate = nn.Sequential(nn.Linear(hidden_dim, 1), )

        self.use_prompt = use_prompt
        if use_prompt:
            self.prompt_dropout = nn.Dropout(prompt_dropout) if prompt_dropout > 0 else nn.Identity()
            self.prompt_t = nn.Parameter(torch.zeros(1, hidden_dim))
            self.prompt_a = nn.Parameter(torch.zeros(1, hidden_dim))
            self.prompt_v = nn.Parameter(torch.zeros(1, hidden_dim))
            nn.init.trunc_normal_(self.prompt_t, std=0.02)
            nn.init.trunc_normal_(self.prompt_a, std=0.02)
            nn.init.trunc_normal_(self.prompt_v, std=0.02)

        self.out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))

        self.enable_after = int(enable_after)
        self.tau0 = float(tau0)
        self._cur_epoch = 0

    def set_epoch(self, e: int):
        self._cur_epoch = int(e)

    def _attn_enabled(self) -> bool:
        if not self.use_cross_attn:
            return False
        if self.enable_after < 0:
            return True
        return self._cur_epoch >= self.enable_after

    def _current_tau(self) -> float:
        if not self._attn_enabled():
            return self.tau0
        delta = max(0, self._cur_epoch - max(0, self.enable_after))
        return max(0.6, self.tau0 * (0.95 ** delta))

    def forward(self, text: torch.Tensor, audio: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        t = self.t_temp(text)
        a = self.a_temp(audio)
        v = self.v_temp(visual)

        t = self.t_proj(t); a = self.a_proj(a); v = self.v_proj(v)

        if self.use_prompt:
            t = t + self.prompt_dropout(self.prompt_t)
            a = a + self.prompt_dropout(self.prompt_a)
            v = v + self.prompt_dropout(self.prompt_v)

        concat = torch.cat([t, a, v], dim=-1)
        h = self.fuse_pre(concat)
        if self.use_gate:
            g_t, g_a, g_v, g_ta, g_tv, g_av = self.gate(h).chunk(6, dim=-1)
            ta, tv, av = t * a, t * v, a * v
            fused = (g_t * t + g_a * a + g_v * v + g_ta * ta + g_tv * tv + g_av * av) / 3.0
        else:
            fused = (t + a + v) / 3.0

        if self._attn_enabled():
            tau = self._current_tau()
            t_ctx, _ = self.attn_t(t, [a, v])
            ctx_add = self.ctx_proj(t_ctx)
            u = self.ctx_gate(h)
            g_ctx = torch.sigmoid(u / tau)
            fused = fused + g_ctx * ctx_add
            if self.cross_attn_mode == "all":
                a_ctx, _ = self.attn_a(a, [t, v])
                v_ctx, _ = self.attn_v(v, [t, a])
                fused = fused + g_ctx * self.ctx_proj(a_ctx) + g_ctx * self.ctx_proj(v_ctx)

        return self.out(fused)


def apply_modality_dropout(text, audio, visual, p: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if p <= 0:
        return text, audio, visual
    with torch.no_grad():
        mask = torch.rand(3, device=text.device) < p
        t = torch.zeros_like(text) if mask[0] else text
        a = torch.zeros_like(audio) if mask[1] else audio
        v = torch.zeros_like(visual) if mask[2] else visual
        return t, a, v


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logpt = F.log_softmax(logits, dim=1)
        pt = torch.exp(logpt)
        oh = F.one_hot(target, num_classes=logits.size(1)).float()
        logpt = (logpt * oh).sum(dim=1)
        pt = (pt * oh).sum(dim=1)
        focal = (1 - pt).pow(self.gamma) * (-logpt)
        if self.weight is not None:
            alpha = self.weight[target]
            focal = alpha * focal
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def build_cb_weight_from_counts(counts: torch.Tensor, beta: float, device: torch.device) -> torch.Tensor:
    eps = 1e-12
    en = (1 - beta) / (1 - torch.pow(beta, counts.clamp_min(1.0)))
    w = en / (en.sum() + eps) * counts.numel()
    return w.to(device=device, dtype=torch.float32)


# =========================
# Freq/Time mask
# =========================
def time_mask_segments(x: torch.Tensor, max_segments: int = 2, max_ratio: float = 0.2) -> torch.Tensor:
    if x.dim() != 3 or max_segments <= 0 or max_ratio <= 0:
        return x
    B, T, _ = x.shape
    device = x.device
    # [FIX] do masking without autograd
    with torch.no_grad():
        for b in range(B):
            segs = torch.randint(low=0, high=max_segments + 1, size=(1,), device=device).item()
            for _ in range(segs):
                seg_len = max(1, int(T * float(torch.rand(1, device=device).item()) * max_ratio))
                start = torch.randint(0, max(1, T - seg_len + 1), (1,), device=device).item()
                x[b, start:start + seg_len, :] = 0
    return x


def freq_mask_segments(x: torch.Tensor, max_segments: int = 2, max_ratio: float = 0.2) -> torch.Tensor:
    # x: [B, T, F]
    if x.dim() != 3 or max_segments <= 0 or max_ratio <= 0:
        return x
    B, T, F = x.shape
    device = x.device
    # [FIX] do masking without autograd
    with torch.no_grad():
        for b in range(B):
            segs = torch.randint(0, max_segments + 1, (1,), device=device).item()
            for _ in range(segs):
                seg_len = max(1, int(F * float(torch.rand(1, device=device).item()) * max_ratio))
                start = torch.randint(0, max(1, F - seg_len + 1), (1,), device=device).item()
                x[b, :, start:start + seg_len] = 0
    return x


def compute_task_loss(
    task: TaskSpec,
    logits: torch.Tensor,
    y: torch.Tensor,
    label_smoothing: float = 0.05,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    cb_alpha: torch.Tensor = None,
    # ==== 新增：LA-CE ====
    logit_adjust_tau: float = 0.0,
    class_priors: torch.Tensor = None,
) -> torch.Tensor:
    if task.task_type == "regression":
        y_ = y.reshape(y.size(0), -1)[:, :1].float() if y.dim() >= 2 else y.view(-1, 1).float()
        logits_ = logits.view(-1, 1) if logits.dim() == 1 else logits
        return F.mse_loss(logits_, y_)

    if task.task_type == "binary":
        logit_1d = logits.view(-1)
        if y.dim() == 2 and y.size(1) == 2 and _is_one_hot(y):
            target = y[:, 1].view(-1).float()
        else:
            target = torch.clamp(y.view(-1).float(), 0.0, 1.0)
        return F.binary_cross_entropy_with_logits(logit_1d, target)

    # multiclass
    if logits.dim() == 1:
        raise ValueError(f"Multiclass requires logits of shape [B,C], but got {logits.shape}")
    target_idx = to_class_index(y, logits.size(1))
    # Logit-Adjusted CE
    if logit_adjust_tau > 0.0 and class_priors is not None:
        logits = logits - logit_adjust_tau * torch.log(class_priors + 1e-12)

    if use_focal:
        weight = cb_alpha if cb_alpha is not None else task.class_weights
        return FocalLoss(gamma=focal_gamma, weight=weight)(logits, target_idx)

    return F.cross_entropy(
        logits,
        target_idx,
        weight=task.class_weights,
        label_smoothing=float(label_smoothing) if label_smoothing and label_smoothing > 0 else 0.0,
    )


def compute_proxy_reg_metrics_for_cls(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    # （保留：用于某些需要离散 proxy 的实验场景；回归验证阶段不再使用它作为主日志）
    probs = torch.softmax(logits, dim=1)
    num_classes = probs.size(1)
    idx = torch.arange(num_classes, dtype=probs.dtype, device=probs.device).unsqueeze(0)
    pred_score = (probs * idx).sum(dim=1)
    true_idx = to_class_index(labels.to(probs.device), num_classes).float()
    mae = torch.mean(torch.abs(pred_score - true_idx)).item()
    vx = pred_score - pred_score.mean()
    vy = true_idx - true_idx.mean()
    corr = (vx * vy).sum().item() / (torch.sqrt((vx ** 2).sum() + 1e-9) * torch.sqrt((vy ** 2).sum() + 1e-9) + 1e-9)
    return {"mae": float(mae), "corr": float(corr)}


def try_make_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_csv_header(csv_path: str):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "task_loss", "diff_loss", "val_mae", "val_corr", "val_acc", "val_f1", "lr", "lambda"])


def append_csv_row(csv_path: str, row: List[Union[int, float, str]]):
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(row)


def _collect_logits_labels(model, loader, device, task):
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            x_list, y, _ = extract_batch(batch)
            x_list = move_to_device_list(x_list, device)
            text, audio, visual = x_list[:3]
            out = model(text, audio, visual)
            logits_all.append(out.detach().cpu())
            labels_all.append(y.detach().cpu())
    logits = torch.cat(logits_all, 0)
    labels = torch.cat(labels_all, 0)
    if task.task_type == "multiclass":
        target = to_class_index(labels, logits.size(1))
        return logits, target
    elif task.task_type == "binary":
        if labels.dim() == 2 and labels.size(1) == 2 and _is_one_hot(labels):
            target = labels[:, 1].view(-1).float()
        else:
            target = torch.clamp(labels.view(-1).float(), 0.0, 1.0)
        return logits.view(-1), target
    else:
        return logits, labels


def learn_temperature_on_val(model, val_loader, device, task, lr=0.05, max_iter=100):
    if task.task_type == "regression":
        return None
    logits, target = _collect_logits_labels(model, val_loader, device, task)
    T = torch.ones(1, requires_grad=True, device=device)
    opt = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

    def _closure():
        opt.zero_grad()
        if task.task_type == "multiclass":
            loss = F.cross_entropy(logits.to(device) / T.clamp_min(1e-3), target.to(device))
        else:
            loss = F.binary_cross_entropy_with_logits(logits.to(device) / T.clamp_min(1e-3), target.to(device))
        loss.backward()
        return loss

    try:
        opt.step(_closure)
    except Exception:
        opt = torch.optim.Adam([T], lr=lr)
        for _ in range(max_iter):
            loss = _closure()
            opt.step()
    return float(T.detach().clamp_min(1e-3).cpu())


def learn_class_bias_on_val(
    model, val_loader, device, task,
    steps: int = 200,
    clip: float = 0.4,
    l2: float = 1e-4,
    tol: float = 1e-4,
):
    if task.task_type != "multiclass":
        return None
    from sklearn.metrics import f1_score
    logits_t, y_true_t = _collect_logits_labels(model, val_loader, device, task)
    logits_np = logits_t.numpy()
    y_np = y_true_t.numpy()
    C = logits_np.shape[1]
    b = np.zeros((C,), dtype=np.float32)
    deltas = np.array([-0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5], dtype=np.float32)

    def objective(bvec: np.ndarray) -> float:
        pred = np.argmax(logits_np + bvec[None, :], axis=1)
        f1 = f1_score(y_np, pred, average="weighted")
        return float(f1 - l2 * float((bvec ** 2).sum()))

    best_obj = objective(b)
    for _ in range(int(steps)):
        improved = False
        sweep_gain = 0.0
        for c in range(C):
            cur_best = best_obj
            cur_val = b[c]
            best_bc = cur_val
            for d in deltas:
                bc_try = float(np.clip(cur_val + d, -clip, clip))
                if abs(bc_try - cur_val) < 1e-8:
                    continue
                b_try = b.copy()
                b_try[c] = bc_try
                obj = objective(b_try)
                if obj > cur_best + 1e-12:
                    cur_best = obj
                    best_bc = bc_try
            if best_bc != cur_val:
                b[c] = best_bc
                sweep_gain += (cur_best - best_obj)
                best_obj = cur_best
                improved = True
        if not improved or sweep_gain < tol:
            break
    return torch.from_numpy(b)


def set_dropout_train_bn_eval(module: nn.Module):
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.train()
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def build_optimizer_with_groups(
    model: nn.Module,
    cond_diff: nn.Module,
    base_lr: float,
    diff_lr: float,
    wd_main: float,
    wd_diff: float,
) -> AdamW:
    def is_no_decay(n, p):
        if not p.requires_grad: return True
        if n.endswith(".bias"): return True
        if "LayerNorm.weight" in n or "layer_norm" in n or "ln" in n: return True
        if p.dim() == 1: return True
        return False

    main_decay, main_nod, diff_decay, diff_nod = [], [], [], []
    for n, p in model.named_parameters():
        (main_nod if is_no_decay(n, p) else main_decay).append(p)
    for n, p in cond_diff.named_parameters():
        (diff_nod if is_no_decay(n, p) else diff_decay).append(p)

    return AdamW(
        [
            {"params": main_decay, "lr": base_lr, "weight_decay": wd_main},
            {"params": main_nod,   "lr": base_lr, "weight_decay": 0.0},
            {"params": diff_decay, "lr": diff_lr, "weight_decay": wd_diff},
            {"params": diff_nod,   "lr": diff_lr, "weight_decay": 0.0},
        ], lr=base_lr
    )


@torch.no_grad()
def _val_pred_vector(model, loader, device, task, space: str = "logit") -> torch.Tensor:
    model.eval()
    outs = []
    for batch in loader:
        x_list, y, _ = extract_batch(batch)
        x_list = move_to_device_list(x_list, device)
        text, audio, visual = x_list[:3]
        z = model(text, audio, visual)
        if task.task_type == "multiclass":
            if space == "prob":
                z = torch.softmax(z, dim=1)
        elif task.task_type == "binary":
            z = z.view(-1, 1)
            if space == "prob":
                z = torch.sigmoid(z)
        outs.append(z.detach().cpu().flatten())
    return torch.cat(outs, dim=0)


def select_topk_diverse(
    top_paths: List[Tuple[float, str]],
    K: int,
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    task: TaskSpec,
    ensemble_space: str = "logit",
    alpha: float = 0.2,
) -> List[Tuple[float, str]]:
    if len(top_paths) <= K:
        return top_paths
    M = max(K * 3, K + 2)
    cand = sorted(top_paths, key=lambda x: x[0], reverse=True)[:M]
    reps = []
    for _, p in cand:
        ckpt = torch.load(p, map_location=device)
        state_model = ckpt.get("model", ckpt)
        model.load_state_dict(state_model, strict=False)
        if hasattr(model, "set_epoch"):
            model.set_epoch(9999)
        v = _val_pred_vector(model, val_loader, device, task, space=ensemble_space)
        v = v / (v.norm() + 1e-9)
        reps.append(v)

    selected = [0]
    selected_set = {0}
    while len(selected) < K:
        best_j, best_score = None, -1e9
        for j in range(len(cand)):
            if j in selected_set:
                continue
            sims = []
            vj = reps[j]
            for i in selected:
                vi = reps[i]
                sims.append(float(torch.dot(vj, vi)))
            mean_sim = float(np.mean(sims)) if sims else 0.0
            raw_score = cand[j][0]
            adj = raw_score - alpha * mean_sim
            if adj > best_score:
                best_score = adj
                best_j = j
        selected.append(best_j)
        selected_set.add(best_j)
    return [cand[i] for i in selected]


def train_one_epoch(
    model: nn.Module,
    cond_diff: ConditionalDiffusionModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cur_lambda: float,
    task: TaskSpec,
    modality_dropout_p: float = 0.0,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    cb_alpha: torch.Tensor = None,
    ema_model: AveragedModel = None,
    feat_noise_std: float = 0.0,
    aug_time_mask_p: float = 0.0,
    aug_time_mask_segments: int = 2,
    aug_time_mask_max_ratio: float = 0.2,
    use_amp: bool = False,
    scaler: "torch.amp.GradScaler" = None,
    accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    # ==== 新增：LA-CE + SAM ====
    la_tau: float = 0.0,
    class_priors: torch.Tensor = None,
    use_sam: bool = False,
    sam_rho: float = 0.05,
    label_noise: float = 0.0,
    zero_inputs: bool = False,
    shuffle_labels: bool = False,
    diffusion_enabled: bool = True,
) -> Tuple[float, float, float]:
    model.train()
    cond_diff.train()
    total_loss, total_task, total_diff = 0.0, 0.0, 0.0
    if accum_steps < 1:
        accum_steps = 1
    if use_sam and accum_steps != 1:
        print("[WARN] SAM is best used with accum_steps=1; forcing accum_steps=1")
        accum_steps = 1

    optimizer.zero_grad(set_to_none=True)
    params_all = [p for p in list(model.parameters()) + list(cond_diff.parameters()) if p.requires_grad]

    def _forward_compute_loss(text, audio, visual, y, missing_mode_idx):
        logits = model(text, audio, visual)
        ls = 0.05 if (task.task_type == "multiclass" and not use_focal) else 0.0
        task_loss = compute_task_loss(
            task, logits, y, label_smoothing=ls,
            use_focal=use_focal, focal_gamma=focal_gamma, cb_alpha=cb_alpha,
            logit_adjust_tau=la_tau, class_priors=class_priors
        )
        diff_loss = torch.tensor(0.0, device=text.device)
        if diffusion_enabled and cur_lambda > 0:
            try:
                diff_loss = cond_diff.get_diffusion_loss(text, audio, visual, missing_mode_idx)
                if not torch.is_tensor(diff_loss):
                    diff_loss = torch.tensor(float(diff_loss), device=device)
            except Exception as e:
                print("[WARN] diffusion loss skipped:", e)
                diff_loss = torch.tensor(0.0, device=device)
        loss = task_loss + cur_lambda * diff_loss
        return loss, task_loss, diff_loss

    for step, batch in enumerate(loader):
        x_list, y, missing_mode = extract_batch(batch)
        x_list = move_to_device_list(x_list, device)
        text, audio, visual = x_list[:3]
        y = y.to(device, non_blocking=True)
        y = corrupt_labels(y, task, label_noise, device)
        missing_mode_idx = mask_to_index(missing_mode.to(device, non_blocking=True))

        # 模态 dropout
        text, audio, visual = apply_modality_dropout(text, audio, visual, modality_dropout_p)

        # 时域/频域遮罩增强（音频）
        if aug_time_mask_p > 0 and torch.rand(1, device=device).item() < aug_time_mask_p and audio.dim() == 3:
            audio = time_mask_segments(audio, aug_time_mask_segments, aug_time_mask_max_ratio)
        if aug_time_mask_p > 0 and torch.rand(1, device=device).item() < aug_time_mask_p and audio.dim() == 3:
            audio = freq_mask_segments(audio, aug_time_mask_segments, aug_time_mask_max_ratio)

        # 特征噪声
        if feat_noise_std > 0:
            audio = audio + torch.randn_like(audio) * feat_noise_std
            visual = visual + torch.randn_like(visual) * feat_noise_std
        if zero_inputs:
            text = torch.zeros_like(text)
            audio = torch.zeros_like(audio)
            visual = torch.zeros_like(visual)

        if not use_sam:
            # ===== 普通一步优化 =====
            if use_amp and scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss, task_loss, diff_loss = _forward_compute_loss(text, audio, visual, y, missing_mode_idx)
                    loss_scaled = loss / accum_steps
                scaler.scale(loss_scaled).backward()
            else:
                loss, task_loss, diff_loss = _forward_compute_loss(text, audio, visual, y, missing_mode_idx)
                (loss / accum_steps).backward()

            if (step + 1) % accum_steps == 0:
                if use_amp and scaler is not None:
                    if max_grad_norm and max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(params_all, max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if max_grad_norm and max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(params_all, max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema_model is not None:
                    ema_model.update_parameters(model)

        else:
            # ===== SAM 两步优化 =====
            # (1) 正常前向/反传
            if use_amp and scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss, task_loss, diff_loss = _forward_compute_loss(text, audio, visual, y, missing_mode_idx)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss, task_loss, diff_loss = _forward_compute_loss(text, audio, visual, y, missing_mode_idx)
                loss.backward()

            # 计算 e_w 并加扰动
            grads = [p.grad for p in params_all if p.grad is not None]
            if len(grads) == 0:
                grad_norm = torch.tensor(0.0, device=device)
            else:
                grad_norm = torch.norm(torch.stack([g.norm() for g in grads]), p=2)
            scale = sam_rho / (grad_norm + 1e-12)
            e_ws = []
            for p in params_all:
                if p.grad is None:
                    e_ws.append(None); continue
                e_w = p.grad * scale
                p.add_(e_w)
                e_ws.append(e_w)

            optimizer.zero_grad(set_to_none=True)

            # (2) 扰动后再次前向/反传
            if use_amp and scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss2, _, _ = _forward_compute_loss(text, audio, visual, y, missing_mode_idx)
                scaler.scale(loss2).backward()
                scaler.unscale_(optimizer)
                if max_grad_norm and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params_all, max_grad_norm)
            else:
                loss2, _, _ = _forward_compute_loss(text, audio, visual, y, missing_mode_idx)
                loss2.backward()
                if max_grad_norm and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params_all, max_grad_norm)

            # 还原权重，再 step
            for p, e_w in zip(params_all, e_ws):
                if e_w is not None:
                    p.sub_((e_w))

            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            if ema_model is not None:
                ema_model.update_parameters(model)

        total_loss += float(loss.detach().item())
        total_task += float(task_loss.detach().item())
        total_diff += float(diff_loss.detach().item())

    n = max(len(loader), 1)
    return total_loss / n, total_task / n, total_diff / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    task: TaskSpec,
) -> Dict[str, float]:
    model.eval()
    preds, labels = [], []
    for batch in loader:
        x_list, y, _ = extract_batch(batch)
        x_list = move_to_device_list(x_list, device)
        text, audio, visual = x_list[:3]
        y = y.to(device, non_blocking=True)
        logits = model(text, audio, visual)
        preds.append(logits.detach().cpu())
        labels.append(y.detach().cpu())

    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)

    if task.task_type == "regression":
        return compute_metrics(preds, labels)

    from sklearn.metrics import accuracy_score, f1_score

    if task.task_type == "binary":
        prob = torch.sigmoid(preds.view(-1))
        y_pred = (prob >= 0.5).int().numpy()
        if labels.dim() == 2 and labels.size(1) == 2 and _is_one_hot(labels):
            y_true = labels[:, 1].view(-1).int().numpy()
        else:
            y_true = torch.clamp(labels.view(-1), 0, 1).int().numpy()
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        return {"acc": float(acc), "f1": float(f1)}

    y_pred = preds.argmax(dim=1).numpy()
    y_true = to_class_index(labels, preds.size(1)).numpy()
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return {"acc": float(acc), "f1": float(f1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--diffusion_lr", type=float, default=3e-5)
    parser.add_argument("--diffusion_loss_weight", type=float, default=0.25)
    parser.add_argument("--lambda_min", type=float, default=0.20)
    parser.add_argument("--diffusion_timesteps", type=int, default=150)
    parser.add_argument("--num_epochs", type=int, default=150)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--drop_rate", type=float, default=0.25)

    parser.add_argument("--temporal_encoder", type=str, choices=["none", "bilstm"], default="bilstm")
    parser.add_argument("--temporal_hidden", type=int, default=128)
    parser.add_argument("--disable_bilstm", action="store_true", help="Force temporal encoder to mean pooling")

    parser.add_argument("--use_cross_attn", action="store_true")
    parser.add_argument("--cross_attn_mode", type=str, choices=["t2av", "all"], default="t2av")
    parser.add_argument("--enable_cross_attn_after", type=int, default=-1)
    parser.add_argument("--cross_attn_tau", type=float, default=1.2)
    parser.add_argument("--disable_cross_attn", action="store_true", help="Convenience flag to turn off cross attention")

    parser.add_argument("--warmup_epochs", type=int, default=8)
    parser.add_argument("--cosine_tmax", type=int, default=-1)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--early_stop_min_epochs", type=int, default=15)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_v9_temporal_xattn_logitens")

    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--ens_temp", type=float, default=1.0)
    parser.add_argument("--ensemble_space", type=str, choices=["logit", "prob"], default="logit")

    parser.add_argument("--freeze_epochs", type=int, default=10)
    parser.add_argument("--ramp_epochs", type=int, default=20)
    parser.add_argument("--stop_diffusion_epoch", type=int, default=-1)
    parser.add_argument("--disable_diffusion", action="store_true", help="Disable diffusion branch and loss")
 
    parser.add_argument("--eval_interval", type=int, default=1)

    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--use_focal", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--use_cb_focal", action="store_true")
    parser.add_argument("--cb_beta", type=float, default=0.999)
    parser.add_argument("--modality_dropout_p", type=float, default=0.0)

    parser.add_argument("--feat_noise_std", type=float, default=0.0)

    parser.add_argument("--use_swa", action="store_true")
    parser.add_argument("--swa_start_epoch", type=int, default=999999)
    parser.add_argument("--swa_freq", type=int, default=1)

    parser.add_argument("--use_temp_scaling", action="store_true")
    parser.add_argument("--ts_lr", type=float, default=0.05)
    parser.add_argument("--ts_max_iter", type=int, default=100)
    parser.add_argument("--use_class_bias", action="store_true")

    parser.add_argument("--aug_time_mask_p", type=float, default=0.0)
    parser.add_argument("--aug_time_mask_segments", type=int, default=2)
    parser.add_argument("--aug_time_mask_max_ratio", type=float, default=0.2)

    parser.add_argument("--wd_main", type=float, default=1e-4)
    parser.add_argument("--wd_diff", type=float, default=5e-5)

    parser.add_argument("--score_metric", type=str, default="f1", choices=["f1", "acc", "mix"])
    parser.add_argument("--mcdo_passes", type=int, default=1)
    parser.add_argument("--label_noise", type=float, default=0.0, help="Fraction of labels to corrupt during training")
    parser.add_argument("--zero_inputs", action="store_true", help="Zero out inputs during training (for ablation)")
    parser.add_argument("--shuffle_labels", action="store_true", help="Shuffle labels every batch (for ablation)")

    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--val_log_style", type=str, choices=["auto", "cls", "reg"], default="auto")

    # ==== 新增：LA-CE 与 SAM ====
    parser.add_argument("--use_la", action="store_true", help="Use Logit-Adjusted CE")
    parser.add_argument("--la_tau", type=float, default=1.2, help="Temperature for logit adjustment")

    parser.add_argument("--use_sam", action="store_true", help="Use SAM optimizer steps")
    parser.add_argument("--sam_rho", type=float, default=0.05, help="Neighborhood size for SAM")

    parser.add_argument("--disable_gate", action="store_true", help="Remove gated fusion, use plain averaging")
    parser.add_argument("--disable_prompt", action="store_true", help="Remove learnable prompts")
    parser.add_argument("--prompt_dropout", type=float, default=0.05, help="Dropout applied to prompts")

    parser.add_argument("--fold_idx", type=int, default=-1,
                        help="0..4 表示 IEMOCAP 第几折；-1 表示不指定，让数据加载器自行处理")
    parser.add_argument("--metrics_out", type=str, default="",
                        help="最终测试指标保存路径（json）。为空则不保存")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    loaders_ret = get_loader(args)
    if isinstance(loaders_ret, tuple):
        if len(loaders_ret) >= 4 and isinstance(loaders_ret[0], dict):
            dataloaders = loaders_ret[0]
            input_dims = loaders_ret[1]
        elif len(loaders_ret) >= 3 and all(hasattr(x, "__iter__") for x in loaders_ret[:3]):
            dataloaders = {"train": loaders_ret[0], "valid": loaders_ret[1], "test": loaders_ret[2]}
            input_dims = (300, 5, 20)
        else:
            raise ValueError("Unexpected return structure from get_loader(args).")
    elif isinstance(loaders_ret, dict):
        dataloaders = loaders_ret
        input_dims = (300, 5, 20)
    else:
        raise ValueError("Unexpected return type from get_loader(args).")

    train_loader = dataloaders["train"]
    val_loader = dataloaders.get("valid") or dataloaders.get("val")
    test_loader = dataloaders["test"]
    if val_loader is None:
        raise ValueError("Validation loader not found.")

    task, counts = infer_task_spec(args.dataset, train_loader, device)
    xlist_dbg, y_dbg, _ = extract_batch(next(iter(train_loader)))
    print(f"[INFO] Detected task: {task} | label_shape={tuple(y_dbg.shape)}")
    print(f"Debug: Initializing CrossModalDiffusion with dims: text={input_dims[0]}, audio={input_dims[1]}, visual={input_dims[2]}")

    temporal_enc = "none" if args.disable_bilstm else args.temporal_encoder
    use_cross_attn = args.use_cross_attn and (not args.disable_cross_attn)
    use_gate = not args.disable_gate
    use_prompt = not args.disable_prompt
    diffusion_enabled_global = (not args.disable_diffusion) and (args.diffusion_loss_weight > 0)

    out_dim = task.num_classes if task.task_type == "multiclass" else 1
    model = CrossModalFusionHead(
        in_dims=input_dims[:3],
        hidden_dim=args.hidden_dim,
        out_dim=out_dim,
        dropout=args.drop_rate,
        temporal_encoder=temporal_enc,
        temporal_hidden=args.temporal_hidden,
        use_cross_attn=use_cross_attn,
        cross_attn_mode=args.cross_attn_mode,
        enable_after=args.enable_cross_attn_after,
        tau0=args.cross_attn_tau,
        use_gate=use_gate,
        use_prompt=use_prompt,
        prompt_dropout=args.prompt_dropout,
    ).to(device)

    cond_diff = ConditionalDiffusionModel(
        text_dim=input_dims[0],
        audio_dim=input_dims[1],
        visual_dim=input_dims[2],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_timesteps=args.diffusion_timesteps,
    ).to(device)
    if not diffusion_enabled_global:
        cond_diff.eval()
        cond_diff.requires_grad_(False)

    print(f"[INFO] Model output_dim={out_dim} for task_type={task.task_type}")
    print(f"[INFO] Model ready. Training for {args.num_epochs} epochs...")

    optimizer = build_optimizer_with_groups(
        model, cond_diff, base_lr=args.lr, diff_lr=args.diffusion_lr, wd_main=args.wd_main, wd_diff=args.wd_diff
    )

    warmup_epochs = max(0, int(args.warmup_epochs))
    cosine_tmax = args.cosine_tmax if args.cosine_tmax > 0 else max(1, args.num_epochs - warmup_epochs)
    sched1 = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(1, warmup_epochs))
    sched2 = CosineAnnealingLR(optimizer, T_max=cosine_tmax)
    scheduler = SequentialLR(optimizer, schedulers=[sched1, sched2], milestones=[warmup_epochs])

    ema_model = AveragedModel(model) if args.use_ema else None

    cb_alpha = None
    if args.use_cb_focal and task.task_type == "multiclass" and counts is not None:
        cb_alpha = build_cb_weight_from_counts(counts.to(device), beta=args.cb_beta, device=device)
        print(f"[INFO] Using Class-Balanced alpha (beta={args.cb_beta}): {cb_alpha.detach().cpu().tolist()}")

    swa_model = AveragedModel(model) if args.use_swa else None
    swa_scheduler = None
    swa_start = int(args.swa_start_epoch)

    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.use_amp))

    try_make_dir(args.ckpt_dir)
    csv_path = "training_log.csv"
    save_csv_header(csv_path)

    best_for_earlystop = -1e9
    no_improve = 0

    lam_target = float(args.diffusion_loss_weight if diffusion_enabled_global else 0.0)
    freeze_epochs = int(args.freeze_epochs)
    ramp_epochs = int(args.ramp_epochs)

    def diffusion_lambda(epoch: int) -> float:
        if epoch < freeze_epochs:
            return 0.0
        t = min(1.0, (epoch - freeze_epochs) / max(1, ramp_epochs))
        return t * lam_target

    top_paths: List[Tuple[float, str]] = []

    # ==== 新增：训练集先验（LA-CE 用）====
    class_priors = None
    if task.task_type == "multiclass" and counts is not None:
        class_priors = (counts / counts.sum().clamp_min(1.0)).to(device=device, dtype=torch.float32)

    for epoch in range(args.num_epochs):
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch)

        if args.use_swa and epoch == swa_start:
            swa_scheduler = SWALR(optimizer, swa_lr=args.lr * 0.5)

        if diffusion_enabled_global and args.stop_diffusion_epoch >= 0 and epoch == args.stop_diffusion_epoch:
            for p in cond_diff.parameters():
                p.requires_grad = False
            for g in optimizer.param_groups:
                if g.get("lr", 0.0) == args.diffusion_lr:
                    g["lr"] = 0.0
            print(f"[INFO] Freeze diffusion branch at epoch {epoch}")

        cur_lambda = diffusion_lambda(epoch) if diffusion_enabled_global else 0.0
        diff_enabled_epoch = diffusion_enabled_global and (cur_lambda > 0.0) and (
            args.stop_diffusion_epoch < 0 or epoch < args.stop_diffusion_epoch
        )

        mdrop_base = args.modality_dropout_p
        if args.num_epochs > 0:
            decay_t = max(0.0, min(1.0, (epoch - int(0.6 * args.num_epochs)) / max(1, int(0.3 * args.num_epochs))))
        else:
            decay_t = 1.0
        mdrop_now = mdrop_base * (1.0 - decay_t)

        tr_loss, tr_task, tr_diff = train_one_epoch(
            model=model, cond_diff=cond_diff, loader=train_loader, optimizer=optimizer, device=device,
            cur_lambda=cur_lambda, task=task, modality_dropout_p=mdrop_now, use_focal=args.use_focal,
            focal_gamma=args.focal_gamma, cb_alpha=cb_alpha, ema_model=ema_model, feat_noise_std=args.feat_noise_std,
            aug_time_mask_p=args.aug_time_mask_p, aug_time_mask_segments=args.aug_time_mask_segments,
            aug_time_mask_max_ratio=args.aug_time_mask_max_ratio, use_amp=args.use_amp, scaler=scaler,
            accum_steps=max(1, int(args.accum_steps)), max_grad_norm=args.max_grad_norm,
            # 新增参数
            la_tau=(args.la_tau if args.use_la else 0.0),
            class_priors=class_priors,
            use_sam=args.use_sam,
            sam_rho=args.sam_rho,
            label_noise=float(args.label_noise),
            diffusion_enabled=diff_enabled_epoch,
        )

        eval_model = ema_model if ema_model is not None else model

        if swa_model is not None and epoch >= swa_start and ((epoch - swa_start) % max(1, args.swa_freq) == 0):
            swa_model.update_parameters(model)

        do_eval = (epoch % max(1, args.eval_interval) == 0) or (epoch == args.num_epochs - 1)
        lr_now = optimizer.param_groups[0]["lr"]

        if do_eval:
            raw_metrics = evaluate(
                eval_model,
                loader=val_loader,
                device=device,
                task=task,
            )

            # [FIX] —— 统一用 true regression 指标（而不是 proxy）
            if task.task_type == "regression" or (args.val_log_style == "reg"):
                val_mae = float(raw_metrics.get("mae", 0.0))
                val_corr = float(raw_metrics.get("corr", 0.0))
                print(f"Epoch {epoch} | loss {tr_loss:.4f} (task {tr_task:.4f}, diff {tr_diff:.4f}) | val MAE {val_mae:.4f}, Corr {val_corr:.4f}")
                val_acc = None
                val_f1 = None
            else:
                val_acc = raw_metrics.get("acc")
                val_f1 = raw_metrics.get("f1")
                print(f"Epoch {epoch} | loss {tr_loss:.4f} (task {tr_task:.4f}, diff {tr_diff:.4f}) | val Acc {val_acc if val_acc is not None else 0:.4f}, F1 {val_f1 if val_f1 is not None else 0:.4f}")
                val_mae = None
                val_corr = None

            append_csv_row(
                csv_path,
                [
                    epoch,
                    float(tr_loss),
                    float(tr_task),
                    float(tr_diff),
                    float(val_mae) if val_mae is not None else "",
                    float(val_corr) if val_corr is not None else "",
                    float(val_acc) if val_acc is not None else "",
                    float(val_f1) if val_f1 is not None else "",
                    float(lr_now),
                    float(cur_lambda),
                ],
            )

            # [FIX] —— 回归场景下用 -MAE 做 early-stop/保存评分（越大越好）
            score = None
            if task.task_type == "regression" and (val_mae is not None):
                score = -val_mae
            elif task.task_type != "regression":
                if args.score_metric == "f1" and val_f1 is not None:
                    score = float(val_f1)
                elif args.score_metric == "acc" and val_acc is not None:
                    score = float(val_acc)
                elif args.score_metric == "mix" and (val_f1 is not None) and (val_acc is not None):
                    score = 0.7 * float(val_f1) + 0.3 * float(val_acc)
                elif val_f1 is not None:
                    score = float(val_f1)
                elif val_acc is not None:
                    score = float(val_acc)

            if score is not None:
                ckpt_path = os.path.join(args.ckpt_dir, f"epoch_{epoch}_score_{score:.4f}.pt")
                state_dict_for_save = eval_model.state_dict() if eval_model is not None else model.state_dict()
                torch.save({"model": state_dict_for_save, "diff": cond_diff.state_dict(), "score": score}, ckpt_path)
                top_paths.append((score, ckpt_path))
                top_paths = sorted(top_paths, key=lambda x: x[0], reverse=True)[: args.topk]

                if score > best_for_earlystop + 1e-6:
                    best_for_earlystop = score
                    no_improve = 0
                else:
                    no_improve += 1

                if args.early_stop_patience > 0 and epoch + 1 >= max(args.early_stop_min_epochs, 1):
                    if no_improve >= args.early_stop_patience:
                        print(f"[EarlyStopping] Stopped early at epoch {epoch}")
                        break
        else:
            print(f"Epoch {epoch} | loss {tr_loss:.4f} (task {tr_task:.4f}, diff {tr_diff:.4f})")

        if swa_scheduler is not None and epoch >= swa_start:
            swa_scheduler.step()
        else:
            scheduler.step()

    if args.use_swa and swa_model is not None:
        print("[INFO] SWA: updating BN statistics...")
        update_bn(train_loader, swa_model)
        model_for_test = swa_model
    elif args.use_ema and ema_model is not None:
        model_for_test = ema_model
    else:
        model_for_test = model

    print("\n[Testing Top-K Weighted Ensemble]")
    def _dump_metrics(metrics: dict):
        out_path = getattr(args, "metrics_out", "")
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                # 全部转 float，避免 numpy.float32 序列化报错
                metrics_f = {k: float(v) for k, v in metrics.items()}
                json.dump(metrics_f, f, ensure_ascii=False, indent=2)
            print(f"[Saved] {out_path}")

    if not top_paths:
        last_score = best_for_earlystop if best_for_earlystop > -1e8 else 0.0
        last_path = os.path.join("checkpoints_auto", f"final_epoch_score_{last_score:.4f}.pt")
        try_make_dir("checkpoints_auto")
        torch.save({"model": model_for_test.state_dict(), "diff": cond_diff.state_dict(), "score": last_score}, last_path)
        top_paths = [(last_score, last_path)]

    top_paths = select_topk_diverse(
        top_paths=top_paths,
        K=int(args.topk),
        model=model,
        val_loader=val_loader,
        device=device,
        task=task,
        ensemble_space=args.ensemble_space,
        alpha=0.2,
    )

    scores = torch.tensor([s for s, _ in top_paths], dtype=torch.float32)
    if hasattr(model, "set_epoch"):
        model.set_epoch(9999)

    if args.ens_temp >= 9.9:
        weights = (torch.ones_like(scores) / scores.numel()).tolist()
    else:
        weights = torch.softmax(scores / max(1e-6, args.ens_temp), dim=0).tolist()

    labels_stack = None
    accum_logits_or_probs = None

    for (w, (_, path)) in zip(weights, top_paths):
        ckpt = torch.load(path, map_location=device)
        state_model = ckpt.get("model")
        state_diff = ckpt.get("diff")
        if state_model is None:
            state_model = ckpt
        model.load_state_dict(state_model, strict=False)
        # diffusion weights were only used during training; inference ensemble keeps main model only

        T_val = 1.0
        b_val = None
        if (args.use_temp_scaling or args.use_class_bias) and task.task_type != "regression":
            if args.use_temp_scaling:
                T_val = learn_temperature_on_val(model, val_loader, device, task, lr=args.ts_lr, max_iter=args.ts_max_iter)
            if args.use_class_bias and task.task_type == "multiclass":
                b_val = learn_class_bias_on_val(model, val_loader, device, task)

        model.eval()
        per_model_logits_or_probs = []
        per_model_labels = []
        with torch.no_grad():
            for batch in test_loader:
                x_list, y, _ = extract_batch(batch)
                x_list = move_to_device_list(x_list, device)
                text, audio, visual = x_list[:3]
                y = y.to(device, non_blocking=True)

                if args.mcdo_passes > 1:
                    set_dropout_train_bn_eval(model)
                    outs = []
                    for _ in range(args.mcdo_passes):
                        z = model(text, audio, visual)
                        if task.task_type == "regression":
                            outs.append(z)
                        elif task.task_type == "multiclass":
                            z = z / T_val
                            if b_val is not None:
                                z = z + b_val.to(z.device)
                            if args.ensemble_space == "prob":
                                outs.append(torch.softmax(z, dim=1))
                            else:
                                outs.append(z)
                        else:
                            z = z.view(-1) / T_val
                            if args.ensemble_space == "prob":
                                outs.append(torch.sigmoid(z).view(-1, 1))
                            else:
                                outs.append(z.view(-1, 1))
                    out = torch.stack(outs, dim=0).mean(dim=0)
                else:
                    z = model(text, audio, visual)
                    if task.task_type == "regression":
                        out = z
                    elif task.task_type == "multiclass":
                        z = z / T_val
                        if b_val is not None:
                            z = z + b_val.to(z.device)
                        out = torch.softmax(z, dim=1) if args.ensemble_space == "prob" else z
                    else:
                        z = z.view(-1) / T_val
                        out = torch.sigmoid(z).view(-1, 1) if args.ensemble_space == "prob" else z.view(-1, 1)

                per_model_logits_or_probs.append(out.detach().cpu())
                per_model_labels.append(y.detach().cpu())

        per_model = torch.cat(per_model_logits_or_probs, dim=0) * w
        if labels_stack is None:
            labels_stack = torch.cat(per_model_labels, dim=0)

        if accum_logits_or_probs is None:
            accum_logits_or_probs = per_model
        else:
            accum_logits_or_probs = accum_logits_or_probs + per_model

    # ===== 根据任务类型计算最终指标并保存 =====
    if task.task_type == "regression":
        final_preds = accum_logits_or_probs
        reg_metrics = compute_metrics(final_preds, labels_stack)
        from sklearn.metrics import accuracy_score, f1_score
        cls_preds = (final_preds.view(-1) >= 0).int()
        cls_labels = (labels_stack.reshape(labels_stack.size(0), -1)[:, 0] >= 0).int()
        acc2 = accuracy_score(cls_labels.cpu(), cls_preds.cpu())
        f12 = f1_score(cls_labels.cpu(), cls_preds.cpu(), average="weighted")
        print("\n[Final Ensemble Test Results]")
        print(f"MAE: {reg_metrics.get('mae', 0):.4f} | Corr: {reg_metrics.get('corr', 0):.4f} | Acc-2: {acc2:.4f} | F1-2: {f12:.4f}")
        metrics_to_save = {
            "mae": float(reg_metrics.get("mae", 0.0)),
            "corr": float(reg_metrics.get("corr", 0.0)),
            "acc2": float(acc2),
            "f12": float(f12),
        }
        _dump_metrics(metrics_to_save)

    elif task.task_type == "binary":
        from sklearn.metrics import accuracy_score, f1_score
        if args.ensemble_space == "prob":
            y_prob = accum_logits_or_probs.view(-1)
        else:
            y_prob = torch.sigmoid(accum_logits_or_probs.view(-1))
        y_pred = (y_prob >= 0.5).int().numpy()
        if labels_stack.dim() == 2 and labels_stack.size(1) == 2 and _is_one_hot(labels_stack):
            y_true = labels_stack[:, 1].view(-1).int().numpy()
        else:
            y_true = torch.clamp(labels_stack.view(-1), 0, 1).int().numpy()
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        print("\n[Final Ensemble Test Results]")
        print(f"Acc: {acc:.4f} | F1: {f1:.4f}")
        metrics_to_save = {"acc": float(acc), "f1": float(f1)}
        _dump_metrics(metrics_to_save)

    else:  # multiclass
        from sklearn.metrics import accuracy_score, f1_score
        if args.ensemble_space == "prob":
            probs = accum_logits_or_probs
        else:
            probs = torch.softmax(accum_logits_or_probs, dim=1)
        y_pred = probs.argmax(dim=1).numpy()
        y_true = to_class_index(labels_stack, probs.size(1)).numpy()
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        print("\n[Final Ensemble Test Results]")
        print(f"Acc: {acc:.4f} | F1: {f1:.4f}")
        metrics_to_save = {"acc": float(acc), "f1": float(f1)}
        _dump_metrics(metrics_to_save)


if __name__ == "__main__":
    main()
