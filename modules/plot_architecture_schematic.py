"""
Quick schematic of the Diffusion-MPLMM pipeline.

The figure highlights:
- Inputs with missing-modality indicator
- CrossModalDiffusion generator (fills missing modalities)
- Three prompt types (modality-signal, missing-type, cross-modal prompts)
- MulT-style cross-modal + self-attention backbone
- Fusion head and loss composition

Output is saved to figs/mplmm_architecture.png.
"""

from pathlib import Path
import argparse
import os
import platform
import subprocess
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def add_box(ax, xy, w, h, text, fc="#e8eaf6", ec="#3f51b5", size=10, lw=1.2):
    """Draw a rounded rectangle and put text at its center."""
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.08,rounding_size=0.05",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        wrap=True,
    )
    return patch


def add_arrow(ax, start, end, text=None, dy=0.0, color="#555", style="-|>", lw=1.4):
    """Draw an arrow from start to end with optional label."""
    ax.annotate(
        text or "",
        xy=end,
        xytext=start,
        textcoords="data",
        arrowprops=dict(arrowstyle=style, color=color, linewidth=lw),
        ha="center",
        va="center",
    )
    if text:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2 + dy
        ax.text(mid_x, mid_y, text, ha="center", va="center", fontsize=9, color=color)


def open_in_viewer(path: Path):
    """Best-effort attempt to open the image with the OS default viewer."""
    try:
        if platform.system() == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:  # pragma: no cover - UX helper
        print(f"[warn] Failed to open viewer automatically: {exc}")


def main(show: bool = False,
         out_path: Path = Path("figs") / "mplmm_architecture.png",
         auto_open: bool = False):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    # Layout grid helpers
    col = {
        "input": 0.2,
        "mask": 1.2,
        "diffusion": 2.4,
        "prompts": 4.2,
        "transformer": 6.2,
        "fusion": 8.4,
        "output": 9.8,
    }
    row = {"text": 3.8, "audio": 2.3, "visual": 0.8}

    # Inputs
    b_txt = add_box(ax, (col["input"], row["text"]), 1.2, 0.8, "Text\nfeature")
    b_aud = add_box(ax, (col["input"], row["audio"]), 1.2, 0.8, "Audio\nfeature")
    b_vis = add_box(ax, (col["input"], row["visual"]), 1.2, 0.8, "Visual\nfeature")
    b_mask = add_box(
        ax,
        (col["mask"], 2.4),
        1.1,
        1.0,
        "Missing\nmode id\nm∈{0…5}",
        fc="#fff3e0",
        ec="#fb8c00",
    )

    # Diffusion generator
    b_diff = add_box(
        ax,
        (col["diffusion"], 1.2),
        1.8,
        2.6,
        "CrossModal\nDiffusion\n(generate missing)",
        fc="#e3f2fd",
        ec="#2196f3",
    )

    # Prompts block
    b_prompt = add_box(
        ax,
        (col["prompts"], 1.0),
        1.8,
        3.0,
        "Prompts\n- modality-signal\n- missing-type\n- cross-modal",
        fc="#f3e5f5",
        ec="#8e24aa",
    )

    # Transformer backbone
    b_trans = add_box(
        ax,
        (col["transformer"], 1.1),
        1.8,
        2.8,
        "MulT-style\nCross-Attn x6\n+ Self-Attn x3",
        fc="#e8f5e9",
        ec="#43a047",
    )

    # Fusion + head
    b_fuse = add_box(
        ax,
        (col["fusion"], 2.0),
        1.4,
        1.6,
        "Concat +\nResidual MLP",
        fc="#ede7f6",
        ec="#5e35b1",
    )

    # Output
    b_out = add_box(
        ax,
        (col["output"], 1.6),
        0.8,
        1.2,
        "Prediction\n(logits)",
        fc="#fffde7",
        ec="#fbc02d",
    )

    # Arrows from inputs to diffusion and prompts
    add_arrow(ax, (col["input"] + 1.2, row["text"] + 0.4), (col["diffusion"], 3.4))
    add_arrow(ax, (col["input"] + 1.2, row["audio"] + 0.4), (col["diffusion"], 2.7))
    add_arrow(ax, (col["input"] + 1.2, row["visual"] + 0.4), (col["diffusion"], 1.6))
    add_arrow(
        ax,
        (col["mask"] + 1.1, 2.9),
        (col["diffusion"], 3.1),
        text="conditioning",
        dy=0.2,
    )

    # Diffusion to prompts (generated missing modalities)
    add_arrow(
        ax,
        (col["diffusion"] + 1.8, 2.5),
        (col["prompts"], 3.5),
        text="filled\nfeatures",
        dy=0.1,
    )

    # Prompts to transformer
    add_arrow(ax, (col["prompts"] + 1.8, 2.5), (col["transformer"], 3.0), text="prompted\nsequences")

    # Direct path for non-missing parts
    add_arrow(
        ax,
        (col["input"] + 1.2, row["audio"] + 0.1),
        (col["prompts"], 1.8),
        text="available\nmodalities",
        dy=-0.2,
    )

    # Transformer to fusion
    add_arrow(ax, (col["transformer"] + 1.8, 2.5), (col["fusion"], 2.8))

    # Fusion to output
    add_arrow(ax, (col["fusion"] + 2.0, 2.4), (col["output"], 2.0))

    # Loss note
    ax.text(
        (col["fusion"] + col["diffusion"]) / 2 + 0.6,
        0.3,
        "Training objective: L_total = L_task + λ · L_diffusion",
        ha="center",
        va="center",
        fontsize=10,
        color="#424242",
        bbox=dict(boxstyle="round,pad=0.2", fc="#f5f5f5", ec="#bdbdbd", lw=0.8),
    )

    # Title
    ax.text(
        5.0,
        5.2,
        "Diffusion-MPLMM Architecture (schematic)",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlim(-0.2, 11.4)
    ax.set_ylim(-0.2, 5.6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved schematic to {out_path.resolve()}")

    if show:
        # Non-blocking show to avoid freezing CLI; window will appear if a GUI backend is available.
        plt.show(block=True)
    else:
        plt.close(fig)

    if auto_open and not show:
        open_in_viewer(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Diffusion-MPLMM schematic.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive window (requires GUI backend).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figs") / "mplmm_architecture.png",
        help="Output path for the saved image.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the saved image with the system default viewer (good for headless matplotlib).",
    )
    args = parser.parse_args()

    main(show=args.show, out_path=args.out, auto_open=args.open)
