"""
Quick helper to visualize the average ACC/F1 reported in the missing-modality table
for MOSI, CH-SIMS, MOSEI, and IEMOCAP.

It generates a 2x2 subplot figure where each panel shows bar charts of ACC and F1
for all compared methods. The script is self-contained; adjust the `DATA` list
if you want to update numbers.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Table values copied from the paper snippet (Avg. across six missing cases)
# Columns: dataset, method, ACC, F1
DATA = [
    # MOSI
    ("MOSI", "LB", 64.21, 67.76),
    ("MOSI", "MS", 64.64, 67.80),
    ("MOSI", "MD", 65.04, 68.08),
    ("MOSI", "MCTN", 66.90, 68.32),
    ("MOSI", "MMIN", 70.84, 71.22),
    ("MOSI", "MPMM", 69.48, 70.11),
    ("MOSI", "MPLMM", 72.14, 72.57),
    ("MOSI", "Ours", 76.09, 76.13),
    # CH-SIMS
    ("CH-SIMS", "LB", 70.11, 76.24),
    ("CH-SIMS", "MS", 69.44, 75.65),
    ("CH-SIMS", "MD", 70.38, 76.19),
    ("CH-SIMS", "MCTN", 70.61, 76.32),
    ("CH-SIMS", "MMIN", 71.41, 76.89),
    ("CH-SIMS", "MPMM", 71.26, 76.85),
    ("CH-SIMS", "MPLMM", 72.07, 77.42),
    ("CH-SIMS", "Ours", 73.09, 72.47),
    # MOSEI
    ("MOSEI", "LB", 72.32, 73.83),
    ("MOSEI", "MS", 71.29, 73.36),
    ("MOSEI", "MD", 72.28, 73.82),
    ("MOSEI", "MCTN", 72.85, 73.94),
    ("MOSEI", "MMIN", 73.25, 74.17),
    ("MOSEI", "MPLMM", 73.75, 74.68),
    ("MOSEI", "Ours", 80.01, 80.23),
    # IEMOCAP
    ("IEMOCAP", "LB", 57.74, 57.42),
    ("IEMOCAP", "MS", 58.53, 58.30),
    ("IEMOCAP", "MD", 59.22, 59.31),
    ("IEMOCAP", "MCTN", 59.19, np.nan),   # F1 not reported (dagger in table)
    ("IEMOCAP", "MMIN", 65.47, np.nan),   # F1 not reported (dagger in table)
    ("IEMOCAP", "MPMM", 65.77, 65.37),
    ("IEMOCAP", "MPLMM", 67.42, 67.22),
    ("IEMOCAP", "Ours", 62.69, 62.61),
]

METHOD_ORDER = ["LB", "MS", "MD", "MCTN", "MMIN", "MPMM", "MPLMM", "Ours"]
DATASETS = ["MOSI", "CH-SIMS", "MOSEI", "IEMOCAP"]
COLORS = {"ACC": "#4C72B0", "F1": "#DD8452"}  # Colorblind-friendly pair


def main():
    df = pd.DataFrame(DATA, columns=["dataset", "method", "ACC", "F1"])

    # Prepare long-form data to make plotting easier.
    long_df = df.melt(
        id_vars=["dataset", "method"],
        value_vars=["ACC", "F1"],
        var_name="metric",
        value_name="score",
    )
    long_df["method"] = pd.Categorical(long_df["method"], METHOD_ORDER, ordered=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    axes = axes.ravel()

    for ax, dataset in zip(axes, DATASETS):
        sub = long_df[long_df["dataset"] == dataset].copy()
        sub.sort_values("method", inplace=True)

        acc = sub[sub["metric"] == "ACC"]["score"].to_numpy()
        f1 = sub[sub["metric"] == "F1"]["score"].to_numpy()
        methods = sub[sub["metric"] == "ACC"]["method"].astype(str).tolist()

        x = np.arange(len(methods))

        # Mask NaNs so lines break where values are missing.
        acc_masked = np.ma.masked_invalid(acc)
        f1_masked = np.ma.masked_invalid(f1)

        acc_line, = ax.plot(
            x,
            acc_masked,
            marker="o",
            linewidth=2.0,
            markersize=6,
            color=COLORS["ACC"],
            label="ACC",
        )
        f1_line, = ax.plot(
            x,
            f1_masked,
            marker="s",
            linewidth=2.0,
            markersize=6,
            color=COLORS["F1"],
            label="F1",
        )

        # Add numeric labels near points; bump labels apart to reduce overlap.
        span = sub["score"].dropna().max() - sub["score"].dropna().min()
        base_offset = max(span * 0.015, 0.6)

        for idx, (xi, yi) in enumerate(zip(x, acc)):
            if np.isnan(yi):
                continue
            ax.text(
                xi,
                yi + base_offset,
                f"{yi:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        for idx, (xi, yi) in enumerate(zip(x, f1)):
            if np.isnan(yi):
                continue
            extra = base_offset
            if not np.isnan(acc[idx]) and abs(yi - acc[idx]) < base_offset * 1.2:
                extra = base_offset * 2.2
            ax.text(
                xi + 0.05,
                yi - extra,
                f"{yi:.2f}",
                ha="left",
                va="top",
                fontsize=8,
                color=COLORS["F1"],
            )

        ax.set_title(dataset, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Score")
        ax.set_ylim(55, max(sub["score"].dropna()) + 6)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Shared legend outside the plot grid.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=11)
    #fig.suptitle("Average performance under missing modalities (ACC / F1)", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_dir = Path("figs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "missing_modalities_avg.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {out_file.resolve()}")
    plt.show()  # pop up window after saving


if __name__ == "__main__":
    main()
