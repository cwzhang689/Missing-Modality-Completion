import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ======================================================
# Helper functions
# ======================================================

def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
    """Ensure shape (B, L, D); if (B, D) expand length dim."""
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected 2D or 3D tensor, got {x.dim()}D tensor with shape {tuple(x.shape)}")


def _pool_seq(x_3d: torch.Tensor) -> torch.Tensor:
    """Mean-pool over time dimension for (B, L, D)."""
    return x_3d.mean(dim=1)


# ======================================================
# Diffusion Scheduler
# ======================================================

class DiffusionScheduler:
    """Simple linear beta scheduler for DDPM-style noise steps."""

    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
            1.0 - self.alphas_cumprod
        )

    def add_noise(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)

    def get_previous_sample(self, x_t: torch.Tensor, t: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
        posterior_mean_coef1_t = self.posterior_mean_coef1[t].reshape(-1, 1, 1)
        posterior_mean_coef2_t = self.posterior_mean_coef2[t].reshape(-1, 1, 1)
        posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1)

        predicted_x0 = (x_t - self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1) * predicted_noise) / \
                       self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)

        x_prev = posterior_mean_coef1_t * predicted_x0 + posterior_mean_coef2_t * x_t

        if t[0] > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + torch.sqrt(posterior_variance_t) * noise

        return x_prev


# ======================================================
# Conditional Diffusion (single modality)
# ======================================================

class ModalityConditionalDiffusion(nn.Module):
    """Unimodal conditional diffusion used inside the cross-modal module."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_timesteps: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps

        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.condition_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        layers = [nn.Linear(hidden_dim * 2, hidden_dim)]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            ])
        layers.append(nn.Linear(hidden_dim, input_dim))
        self.noise_predictor = nn.ModuleList(layers)

        self.scheduler = DiffusionScheduler(num_timesteps=num_timesteps)

    def forward(
        self,
        condition: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        condition = _ensure_3d(condition)
        if target is not None:
            target = _ensure_3d(target)

        batch_size, seq_len, _ = condition.shape
        if t is None:
            t = self.scheduler.sample_timesteps(batch_size, condition.device)

        condition_encoded = self.condition_encoder(condition)
        t_emb = self.time_embedding(t.float().unsqueeze(-1))
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)

        combined = torch.cat([condition_encoded, t_emb], dim=-1)
        noise_pred = combined
        for layer in self.noise_predictor:
            noise_pred = layer(noise_pred)

        if self.training and target is not None:
            x_t, noise = self.scheduler.add_noise(target, t)
            return noise_pred, x_t, noise
        return noise_pred

    def generate(self, condition: torch.Tensor, num_inference_steps: int = 50) -> torch.Tensor:
        condition = _ensure_3d(condition)
        batch_size, seq_len, _ = condition.shape
        device = condition.device

        x_t = torch.randn(batch_size, seq_len, self.input_dim, device=device)
        for i in range(num_inference_steps - 1, -1, -1):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            noise_pred = self.forward(condition, t=t)
            x_t = self.scheduler.get_previous_sample(x_t, t, noise_pred)
        return x_t


# ======================================================
# Cross-Modal Diffusion
# ======================================================

class CrossModalDiffusion(nn.Module):
    """Three-way diffusion to reconstruct missing modality given the other two."""

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        visual_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_timesteps: int = 100,
    ):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)

        self.condition_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
        )

        self.text_diffusion = ModalityConditionalDiffusion(hidden_dim, hidden_dim, num_layers, num_timesteps)
        self.audio_diffusion = ModalityConditionalDiffusion(hidden_dim, hidden_dim, num_layers, num_timesteps)
        self.visual_diffusion = ModalityConditionalDiffusion(hidden_dim, hidden_dim, num_layers, num_timesteps)

        self.text_unproj = nn.Linear(hidden_dim, text_dim)
        self.audio_unproj = nn.Linear(hidden_dim, audio_dim)
        self.visual_unproj = nn.Linear(hidden_dim, visual_dim)

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        visual: torch.Tensor,
        missing_mode: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        text = _ensure_3d(text)
        audio = _ensure_3d(audio)
        visual = _ensure_3d(visual)

        batch_size = text.shape[0]
        text_proj = self.text_proj(text)
        audio_proj = self.audio_proj(audio)
        visual_proj = self.visual_proj(visual)

        generated_text, generated_audio, generated_visual = text.clone(), audio.clone(), visual.clone()

        for i in range(batch_size):
            mode = missing_mode[i].item()

            if mode == 0:  # missing text
                audio_pooled = torch.mean(audio_proj[i], dim=0, keepdim=True)
                visual_pooled = torch.mean(visual_proj[i], dim=0, keepdim=True)
                cond = self.condition_fusion(torch.cat([audio_pooled, visual_pooled], dim=-1))
                cond = cond.expand(text_proj.shape[1], -1)
                if training and self.training:
                    _, x_t, _ = self.text_diffusion(cond.unsqueeze(0), text_proj[i].unsqueeze(0))
                    generated_text[i] = self.text_unproj(x_t.squeeze(0))
                else:
                    gen = self.text_diffusion.generate(cond.unsqueeze(0), num_inference_steps=self.text_diffusion.scheduler.num_timesteps)
                    generated_text[i] = self.text_unproj(gen.squeeze(0))

            elif mode == 1:  # missing audio
                text_pooled = torch.mean(text_proj[i], dim=0, keepdim=True)
                visual_pooled = torch.mean(visual_proj[i], dim=0, keepdim=True)
                cond = self.condition_fusion(torch.cat([text_pooled, visual_pooled], dim=-1))
                cond = cond.expand(audio_proj.shape[1], -1)
                if training and self.training:
                    _, x_t, _ = self.audio_diffusion(cond.unsqueeze(0), audio_proj[i].unsqueeze(0))
                    generated_audio[i] = self.audio_unproj(x_t.squeeze(0))
                else:
                    gen = self.audio_diffusion.generate(cond.unsqueeze(0), num_inference_steps=self.audio_diffusion.scheduler.num_timesteps)
                    generated_audio[i] = self.audio_unproj(gen.squeeze(0))

            elif mode == 2:  # missing visual
                text_pooled = torch.mean(text_proj[i], dim=0, keepdim=True)
                audio_pooled = torch.mean(audio_proj[i], dim=0, keepdim=True)
                cond = self.condition_fusion(torch.cat([text_pooled, audio_pooled], dim=-1))
                cond = cond.expand(visual_proj.shape[1], -1)
                if training and self.training:
                    _, x_t, _ = self.visual_diffusion(cond.unsqueeze(0), visual_proj[i].unsqueeze(0))
                    generated_visual[i] = self.visual_unproj(x_t.squeeze(0))
                else:
                    gen = self.visual_diffusion.generate(cond.unsqueeze(0), num_inference_steps=self.visual_diffusion.scheduler.num_timesteps)
                    generated_visual[i] = self.visual_unproj(gen.squeeze(0))

        return generated_text, generated_audio, generated_visual

    def get_diffusion_loss(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        visual: torch.Tensor,
        missing_mode: torch.Tensor,
    ) -> torch.Tensor:

        text = _ensure_3d(text)
        audio = _ensure_3d(audio)
        visual = _ensure_3d(visual)

        batch_size = text.shape[0]
        text_proj = self.text_proj(text)
        audio_proj = self.audio_proj(audio)
        visual_proj = self.visual_proj(visual)

        total_loss, count = 0.0, 0

        for i in range(batch_size):
            mode = missing_mode[i].item()

            if mode == 0:
                audio_pooled = torch.mean(audio_proj[i], dim=0, keepdim=True)
                visual_pooled = torch.mean(visual_proj[i], dim=0, keepdim=True)
                cond = self.condition_fusion(torch.cat([audio_pooled, visual_pooled], dim=-1))
                cond = cond.expand(text_proj.shape[1], -1)
                noise_pred, x_t, noise = self.text_diffusion(cond.unsqueeze(0), text_proj[i].unsqueeze(0))
                total_loss += F.mse_loss(noise_pred, noise)
                count += 1

            elif mode == 1:
                text_pooled = torch.mean(text_proj[i], dim=0, keepdim=True)
                visual_pooled = torch.mean(visual_proj[i], dim=0, keepdim=True)
                cond = self.condition_fusion(torch.cat([text_pooled, visual_pooled], dim=-1))
                cond = cond.expand(audio_proj.shape[1], -1)
                noise_pred, x_t, noise = self.audio_diffusion(cond.unsqueeze(0), audio_proj[i].unsqueeze(0))
                total_loss += F.mse_loss(noise_pred, noise)
                count += 1

            elif mode == 2:
                text_pooled = torch.mean(text_proj[i], dim=0, keepdim=True)
                audio_pooled = torch.mean(audio_proj[i], dim=0, keepdim=True)
                cond = self.condition_fusion(torch.cat([text_pooled, audio_pooled], dim=-1))
                cond = cond.expand(visual_proj.shape[1], -1)
                noise_pred, x_t, noise = self.visual_diffusion(cond.unsqueeze(0), visual_proj[i].unsqueeze(0))
                total_loss += F.mse_loss(noise_pred, noise)
                count += 1

        return total_loss / max(count, 1)


# Alias for external usage
ConditionalDiffusionModel = CrossModalDiffusion
