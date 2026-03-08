#!/usr/bin/env python
"""
Lightweight ablation runner for train.py.

It batches a set of component ablations (diffusion, cross-attn, temporal encoder,
augmentation) and writes per-run logs/metrics plus a summary CSV.

Example:
    python ablation_experiments.py \
        --dataset mosi \
        --data_path ./Archive/mosi_data.pkl \
        --device cuda \
        --num_epochs 60
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "Archive" / "mosei_data.pkl"
# Prefer fast SSD drive if available; otherwise fall back to project-local folder.
_default_out = Path("D:/mplmm_ablation_runs")
DEFAULT_OUTPUT_ROOT = _default_out if _default_out.drive and _default_out.exists() else PROJECT_ROOT / "ablation_runs"


class LiveLossPopup:
    """Tiny helper to live-refresh loss curves from training_log.csv during a run."""

    def __init__(self, csv_path: Path, title: str):
        self.csv_path = Path(csv_path)
        self.title = title
        self.fig = None
        self.ax = None
        self.lines: Dict[str, object] = {}
        self.last_update = 0.0
        self.last_mtime = 0.0
        self.enabled = True
        try:
            import matplotlib.pyplot as plt  # noqa: WPS433 (runtime import to keep dependency optional)
            import pandas as pd  # noqa: WPS433

            self._plt = plt
            self._pd = pd
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Live loss popup disabled: {exc}")
            self.enabled = False

    def maybe_update(self) -> None:
        if not self.enabled:
            return
        now = time.time()
        if now - self.last_update < 1.2:
            return
        if not self.csv_path.exists():
            return
        mtime = self.csv_path.stat().st_mtime
        if mtime <= self.last_mtime and (now - self.last_update) < 5.0:
            return
        self.last_mtime = mtime
        try:
            df = self._pd.read_csv(self.csv_path)
        except Exception:  # noqa: BLE001
            return
        if "epoch" not in df.columns:
            return
        if self.fig is None:
            self._plt.ion()
            self.fig, self.ax = self._plt.subplots()
            self.ax.set_title(self.title)
            self.ax.set_xlabel("epoch")
            self.ax.set_ylabel("loss")
        palette = {
            "train_loss": "steelblue",
            "task_loss": "seagreen",
            "diff_loss": "darkorange",
        }
        x = df["epoch"].values
        for key, color in palette.items():
            if key not in df.columns:
                continue
            y = df[key].values
            line = self.lines.get(key)
            if line is None:
                (line,) = self.ax.plot(x, y, label=key, color=color, linewidth=1.6)
                self.lines[key] = line
            else:
                line.set_data(x, y)
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.legend(loc="upper right")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.last_update = now

    def close(self) -> None:
        if self.enabled and self.fig is not None:
            self._plt.close(self.fig)

# Fixed presets for running four datasets sequentially without changing their tuned hyperparameters.
PRESET_RUNS: List[Dict[str, object]] = [
    {
        "label": "mosi",
        "base_args": {
            "dataset": "mosi",
            "data_path": str(PROJECT_ROOT / "Archive" / "mosi_data.pkl"),
            "batch_size": 32,
            "lr": 3e-4,
            "diffusion_lr": 3e-5,
            "diffusion_loss_weight": 0.25,
            "lambda_min": 0.20,
            "diffusion_timesteps": 150,
            "num_epochs": 120,
            "hidden_dim": 256,
            "num_layers": 4,
            "drop_rate": 0.3,
            "warmup_epochs": 8,
            "early_stop_patience": 0,
            "topk": 10,
            "device": "cuda",
        },
    },
    {
        "label": "iemocap",
        "base_args": {
            "seed": 2025,
            "dataset": "iemocap",
            "data_path": str(PROJECT_ROOT / "Archive" / "iemocap_data.pkl"),
            "batch_size": 16,
            "lr": 2e-4,
            "diffusion_lr": 2e-5,
            "diffusion_loss_weight": 0.01,
            "diffusion_timesteps": 150,
            "num_epochs": 120,
            "hidden_dim": 384,
            "num_layers": 4,
            "drop_rate": 0.5,
            "warmup_epochs": 8,
            "early_stop_patience": 30,
            "early_stop_min_epochs": 70,
            "freeze_epochs": 60,
            "ramp_epochs": 10,
            "stop_diffusion_epoch": 140,
            "eval_interval": 3,
            "use_ema": True,
            "use_focal": True,
            "focal_gamma": 1.5,
            "use_cb_focal": True,
            "cb_beta": 0.999,
            "modality_dropout_p": 0.22,
            "feat_noise_std": 0.03,
            "use_swa": True,
            "swa_start_epoch": 115,
            "swa_freq": 2,
            "use_temp_scaling": True,
            "use_class_bias": True,
            "ts_lr": 0.05,
            "ts_max_iter": 100,
            "topk": 5,
            "ens_temp": 6,
            "mcdo_passes": 6,
            "score_metric": "f1",
            "aug_time_mask_p": 0.35,
            "aug_time_mask_segments": 2,
            "aug_time_mask_max_ratio": 0.2,
            "wd_main": 1.2e-4,
            "wd_diff": 5e-5,
            "use_amp": True,
            "accum_steps": 1,
            "max_grad_norm": 1.0,
            "temporal_encoder": "bilstm",
            "temporal_hidden": 128,
            "use_cross_attn": True,
            "cross_attn_mode": "t2av",
            "enable_cross_attn_after": 60,
            "cross_attn_tau": 1.1,
            "ensemble_space": "logit",
            "val_log_style": "cls",
            "ckpt_dir": "checkpoints_iemocap_push_t2av_k5",
            "device": "cuda",
        },
    },
    {
        "label": "sims",
        "base_args": {
            "seed": 2025,
            "dataset": "sims",
            "data_path": str(PROJECT_ROOT / "Archive" / "sims.pkl"),
            "batch_size": 32,
            "lr": 2e-4,
            "diffusion_lr": 2e-5,
            "diffusion_loss_weight": 0.005,
            "diffusion_timesteps": 150,
            "num_epochs": 120,
            "hidden_dim": 384,
            "num_layers": 4,
            "drop_rate": 0.40,
            "warmup_epochs": 10,
            "early_stop_patience": 15,
            "early_stop_min_epochs": 40,
            "freeze_epochs": 40,
            "ramp_epochs": 15,
            "stop_diffusion_epoch": 100,
            "eval_interval": 2,
            "use_ema": True,
            "modality_dropout_p": 0.20,
            "feat_noise_std": 0.02,
            "use_swa": True,
            "swa_start_epoch": 90,
            "swa_freq": 2,
            "topk": 5,
            "ens_temp": 8,
            "mcdo_passes": 4,
            "aug_time_mask_p": 0.0,
            "use_amp": False,
            "wd_main": 1e-4,
            "wd_diff": 5e-5,
            "accum_steps": 1,
            "max_grad_norm": 1.0,
            "temporal_encoder": "bilstm",
            "temporal_hidden": 128,
            "use_cross_attn": True,
            "cross_attn_mode": "t2av",
            "enable_cross_attn_after": 40,
            "cross_attn_tau": 1.0,
            "ensemble_space": "logit",
            "val_log_style": "reg",
            "ckpt_dir": "checkpoints_sims_bilstm_xattn_nomask",
            "metrics_out": str(PROJECT_ROOT / "results" / "sims_fixed_reg.json"),
            "device": "cuda",
        },
    },
    {
        "label": "mosei",
        "base_args": {
            "dataset": "mosei",
            "data_path": str(PROJECT_ROOT / "Archive" / "mosei_data.pkl"),
            "batch_size": 16,
            "lr": 2e-4,
            "diffusion_lr": 2e-5,
            "diffusion_loss_weight": 0.20,
            "lambda_min": 0.15,
            "diffusion_timesteps": 200,
            "num_epochs": 120,
            "hidden_dim": 256,
            "num_layers": 4,
            "drop_rate": 0.3,
            "warmup_epochs": 8,
            "early_stop_patience": 3,
            "topk": 10,
            "device": "cuda",
        },
    },
]

# Shared base arguments for all runs. Override with CLI flags or per-ablation overrides.
BASE_ARGS: Dict[str, object] = {
    "dataset": "mosei",
    "data_path": str(DEFAULT_DATA_PATH),
    "device": "cuda",
    "batch_size": 16,
    "lr": 2e-4,
    "diffusion_lr": 2e-5,
    "diffusion_loss_weight": 0.20,
    "lambda_min": 0.15,
    "diffusion_timesteps": 200,
    "num_epochs": 120,
    "hidden_dim": 256,
    "num_layers": 4,
    "drop_rate": 0.3,
    "temporal_encoder": "bilstm",
    "temporal_hidden": 128,
    "use_cross_attn": True,
    "cross_attn_mode": "t2av",
    "enable_cross_attn_after": -1,
    "cross_attn_tau": 1.2,
    "warmup_epochs": 8,
    "cosine_tmax": -1,
    "early_stop_patience": 3,
    "early_stop_min_epochs": 15,
    "topk": 10,
    "ens_temp": 1.0,
    "ensemble_space": "logit",
    "freeze_epochs": 10,
    "ramp_epochs": 20,
    "stop_diffusion_epoch": -1,
    "eval_interval": 1,
    "modality_dropout_p": 0.15,
    "feat_noise_std": 0.01,
    "aug_time_mask_p": 0.1,
    "aug_time_mask_segments": 2,
    "aug_time_mask_max_ratio": 0.2,
    "wd_main": 1e-4,
    "wd_diff": 5e-5,
    "score_metric": "f1",
    "use_amp": True,
    "accum_steps": 1,
    "max_grad_norm": 1.0,
}

# Ablations to run. Each entry overrides BASE_ARGS for that run.
EXPERIMENTS: List[Dict[str, object]] = [
    {
        "name": "full",
        "desc": "Full model: diffusion + cross-attn + BiLSTM + gated fusion + prompts",
        "overrides": {},
    },
    {
        "name": "no_prompt_model",
        "desc": "Disable PromptModel / learnable prompts",
        "overrides": {
            "disable_prompt": True,
        },
    },
    {
        "name": "no_cross_modal_diffusion",
        "desc": "Disable CrossModalDiffusion branch and loss",
        "overrides": {
            "disable_diffusion": True,
            "diffusion_loss_weight": 0.0,
            "stop_diffusion_epoch": 0,
        },
    },
    {
        "name": "no_cross_attention",
        "desc": "Disable cross-modal attention fusion",
        "overrides": {
            "use_cross_attn": False,
            "disable_cross_attn": True,
        },
    },
]

SUMMARY_COLUMNS = [
    "name",
    "status",
    "elapsed_min",
    "log_path",
    "metrics_path",
    "cmd",
    "mae",
    "corr",
    "acc",
    "f1",
    "acc2",
    "f12",
    "delta_mae",
    "delta_corr",
    "delta_acc",
    "delta_f1",
    "delta_acc2",
    "delta_f12",
]


@dataclass
class RunConfig:
    python: str
    output_root: Path
    timeout_min: float
    dry_run: bool
    resume: bool
    only: Tuple[str, ...] | None
    skip: Tuple[str, ...] | None
    override_dataset: str | None
    override_data_path: str | None
    override_device: str | None
    override_batch: int | None
    override_epochs: int | None
    run_all_presets: bool
    live_plot: bool


def parse_cli() -> RunConfig:
    parser = argparse.ArgumentParser(description="Run ablations for train.py")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name for all runs")
    parser.add_argument("--data_path", type=str, default=None, help="Override data path for all runs")
    parser.add_argument("--device", type=str, default=None, help="Override device (e.g., cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--only", nargs="+", default=None, help="Run only the named experiments")
    parser.add_argument("--skip", nargs="+", default=None, help="Skip the named experiments")
    parser.add_argument("--timeout_min", type=float, default=90.0, help="Per-run timeout in minutes (<=0 to disable)")
    parser.add_argument("--output_root", type=str, default=None, help="Where to store logs/metrics")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to use")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument("--resume", action="store_true", help="Skip runs with existing metrics.json")
    parser.add_argument("--run_all_presets", action="store_true", help="Run ablations for all four preset datasets sequentially")
    parser.add_argument("--no_live_plot", action="store_true", help="Disable live loss popup while training")
    args = parser.parse_args()

    # If launched with no CLI args, default to "one-click" run of all presets.
    auto_run_all = (len(sys.argv) == 1)
    run_all_flag = bool(args.run_all_presets or auto_run_all)

    dataset_tag = args.dataset or str(BASE_ARGS.get("dataset", "dataset"))
    default_root = DEFAULT_OUTPUT_ROOT / dataset_tag / "ablation_runs"
    output_root = Path(args.output_root) if args.output_root else default_root
    return RunConfig(
        python=args.python,
        output_root=output_root.resolve(),
        timeout_min=float(args.timeout_min),
        dry_run=bool(args.dry_run),
        resume=bool(args.resume),
        only=tuple(args.only) if args.only else None,
        skip=tuple(args.skip) if args.skip else None,
        override_dataset=args.dataset,
        override_data_path=args.data_path,
        override_device=args.device,
        override_batch=args.batch_size,
        override_epochs=args.num_epochs,
        run_all_presets=run_all_flag,
        live_plot=not bool(args.no_live_plot),
    )


def select_experiments(cfg: RunConfig) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    only_set = set(cfg.only) if cfg.only else None
    skip_set = set(cfg.skip) if cfg.skip else set()
    for exp in EXPERIMENTS:
        name = exp["name"]
        if only_set is not None and name not in only_set:
            continue
        if name in skip_set:
            continue
        selected.append(exp)
    return selected


def dict_to_cli(args: Dict[str, object]) -> List[str]:
    cli: List[str] = []
    for k, v in args.items():
        if v is None:
            continue
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                cli.append(flag)
            continue
        cli.extend([flag, str(v)])
    return cli


def safe_run_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def compute_output_root(cfg: RunConfig, base_args: Dict[str, object]) -> Path:
    dataset_tag = str(base_args.get("dataset", "dataset"))
    if cfg.run_all_presets:
        baseline_default = (DEFAULT_OUTPUT_ROOT / str(BASE_ARGS.get("dataset", "dataset")) / "ablation_runs").resolve()
        if cfg.output_root and Path(cfg.output_root).resolve() != baseline_default:
            base_root = Path(cfg.output_root)
        else:
            base_root = DEFAULT_OUTPUT_ROOT
        root = base_root / dataset_tag / "ablation_runs"
        return root.resolve()

    if cfg.output_root:
        root = Path(cfg.output_root)
        if root.name == "ablation_runs" and root.parent.name == dataset_tag:
            return root.resolve()
        if dataset_tag not in {p.name for p in root.parents} and root.name != dataset_tag:
            root = root / dataset_tag / "ablation_runs"
        return root.resolve()

    return (DEFAULT_OUTPUT_ROOT / dataset_tag / "ablation_runs").resolve()


def load_metrics(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: float(v) for k, v in data.items()}
    except Exception:
        return {}


def compute_delta(value: float | None, baseline: float | None, higher_is_better: bool = True) -> float | None:
    if value is None or baseline is None:
        return None
    diff = value - baseline if higher_is_better else baseline - value
    return round(diff, 4)


def attach_deltas(results: Iterable[Dict[str, object]], baseline: Dict[str, object] | None) -> List[Dict[str, object]]:
    metrics = ("mae", "corr", "acc", "f1", "acc2", "f12")
    results_list = list(results)
    for res in results_list:
        for metric in metrics:
            res[f"delta_{metric}"] = compute_delta(
                res.get(metric),
                baseline.get(metric) if baseline else None,
                higher_is_better=(metric != "mae"),
            )
    return results_list


def write_summary(summary_csv: Path, rows: List[Dict[str, object]]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in SUMMARY_COLUMNS})


def write_effect_sizes(output_root: Path, baseline: Dict[str, object] | None, rows: List[Dict[str, object]]) -> Path:
    effect_path = output_root / "effect_sizes.json"
    metrics = ("mae", "corr", "acc", "f1", "acc2", "f12")
    payload = {
        "baseline": {
            "name": baseline.get("name") if baseline else None,
            "metrics": {m: baseline.get(m) if baseline else None for m in metrics},
        },
        "runs": [],
    }
    for row in rows:
        payload["runs"].append(
            {
                "name": row.get("name"),
                "status": row.get("status"),
                "metrics": {m: row.get(m) for m in metrics},
                "delta_vs_baseline": {m: row.get(f"delta_{m}") for m in metrics},
                "log_path": row.get("log_path"),
                "metrics_path": row.get("metrics_path"),
                "cmd": row.get("cmd"),
            }
        )
    with open(effect_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return effect_path


def run_one_experiment(
    idx: int,
    exp: Dict[str, object],
    base_args: Dict[str, object],
    cfg: RunConfig,
    output_root: Path,
) -> Dict[str, object]:
    name = str(exp["name"])
    run_name = f"{idx:02d}_{safe_run_name(name)}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.json"
    log_path = run_dir / "stdout.log"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    if cfg.resume and metrics_path.exists():
        print(f"[SKIP] {name}: metrics already exist at {metrics_path}")
        metrics = {k: round(v, 4) for k, v in load_metrics(metrics_path).items()}
        return {
            "name": name,
            "status": "skipped",
            "elapsed_min": 0.0,
            "log_path": str(log_path),
            "metrics_path": str(metrics_path),
            "cmd": "skipped",
            "mae": metrics.get("mae"),
            "corr": metrics.get("corr"),
            "acc": metrics.get("acc"),
            "f1": metrics.get("f1"),
            "acc2": metrics.get("acc2"),
            "f12": metrics.get("f12"),
        }

    merged = dict(base_args)
    merged.update(exp.get("overrides", {}))
    merged["metrics_out"] = str(metrics_path)
    merged["ckpt_dir"] = str(ckpt_dir)
    merged["data_path"] = str(Path(merged["data_path"]).expanduser().resolve())

    cmd = [cfg.python, str(PROJECT_ROOT / "train.py")] + dict_to_cli(merged)

    if cfg.dry_run:
        print(f"[DRY-RUN] {name}: {' '.join(cmd)}")
        return {
            "name": name,
            "status": "dry_run",
            "elapsed_min": 0.0,
            "log_path": str(log_path),
            "metrics_path": str(metrics_path),
            "cmd": " ".join(cmd),
            "mae": None,
            "corr": None,
            "acc": None,
            "f1": None,
            "acc2": None,
            "f12": None,
        }

    plotter = LiveLossPopup(run_dir / "training_log.csv", title=f"{base_args.get('dataset')} | {name}") if cfg.live_plot else None

    print(f"\n=== [{datetime.now().strftime('%H:%M:%S')}] Running {name}: {exp.get('desc', '')}")
    print(f"Logs: {log_path}")
    status = "ok"
    start = time.time()
    timeout_s = cfg.timeout_min * 60 if cfg.timeout_min > 0 else None

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=run_dir,
        text=True,
    )
    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            for line in iter(proc.stdout.readline, ""):
                tagged = f"[{name}] {line}"
                sys.stdout.write(tagged)
                lf.write(line)
                lf.flush()
                if plotter:
                    plotter.maybe_update()
                if timeout_s and (time.time() - start) > timeout_s:
                    status = "timeout"
                    proc.kill()
                    break
        proc.wait(timeout=5)
    except KeyboardInterrupt:
        status = "interrupted"
        proc.kill()
        raise
    except subprocess.TimeoutExpired:
        status = "timeout"
        proc.kill()
    except Exception as exc:  # noqa: BLE001
        status = f"error:{exc}"
        proc.kill()

    elapsed_min = (time.time() - start) / 60.0
    if status == "ok" and proc.returncode not in (0, None):
        status = f"fail(rc={proc.returncode})"

    metrics = {k: round(v, 4) for k, v in load_metrics(metrics_path).items()}
    if plotter:
        plotter.maybe_update()
        plotter.close()
    return {
        "name": name,
        "status": status,
        "elapsed_min": round(elapsed_min, 3),
        "log_path": str(log_path),
        "metrics_path": str(metrics_path),
        "cmd": " ".join(cmd),
        "mae": metrics.get("mae"),
        "corr": metrics.get("corr"),
        "acc": metrics.get("acc"),
        "f1": metrics.get("f1"),
        "acc2": metrics.get("acc2"),
        "f12": metrics.get("f12"),
    }


def run_suite(base_args: Dict[str, object], cfg: RunConfig) -> None:
    output_root = compute_output_root(cfg, base_args)
    output_root.mkdir(parents=True, exist_ok=True)

    exps = select_experiments(cfg)
    if not exps:
        print("No experiments selected. Use --only or --skip to control selection.")
        return

    summary_csv = output_root / "ablation_summary.csv"
    print(f"\n=== Dataset: {base_args.get('dataset')} ===")
    print(f"Writing summary to {summary_csv} (overwrites after all runs)")

    results: List[Dict[str, object]] = []
    baseline: Dict[str, object] | None = None

    for idx, exp in enumerate(exps, start=1):
        res = run_one_experiment(idx, exp, base_args, cfg, output_root)
        has_metrics = any(res.get(k) is not None for k in ("mae", "corr", "acc", "f1", "acc2", "f12"))
        if exp["name"] == "full" and has_metrics:
            baseline = res
        res = attach_deltas([res], baseline)[0]
        results.append(res)

        metric_payload = {k: res.get(k) for k in ("mae", "corr", "acc", "f1", "acc2", "f12") if res.get(k) is not None}
        delta_payload = {k: res.get(k) for k in ("delta_mae", "delta_corr", "delta_acc", "delta_f1", "delta_acc2", "delta_f12") if res.get(k) is not None}
        print(
            f"[DONE] {res['name']}: status={res['status']}, metrics={metric_payload}"
            + (f", delta_vs_full={delta_payload}" if baseline else "")
        )

    results = attach_deltas(results, baseline)
    write_summary(summary_csv, results)
    effect_path = write_effect_sizes(output_root, baseline, results)

    print(f"Summary CSV: {summary_csv}")
    print(f"Effect sizes: {effect_path}")


def main() -> None:
    cfg = parse_cli()

    if cfg.run_all_presets:
        for preset in PRESET_RUNS:
            base_args = dict(BASE_ARGS)
            base_args.update(preset["base_args"])
            run_suite(base_args, cfg)
        print("\nAll preset datasets finished.")
        return

    base_args = dict(BASE_ARGS)
    if cfg.override_dataset:
        base_args["dataset"] = cfg.override_dataset
    if cfg.override_data_path:
        base_args["data_path"] = cfg.override_data_path
    if cfg.override_device:
        base_args["device"] = cfg.override_device
    if cfg.override_batch:
        base_args["batch_size"] = cfg.override_batch
    if cfg.override_epochs:
        base_args["num_epochs"] = cfg.override_epochs

    run_suite(base_args, cfg)


if __name__ == "__main__":
    main()
