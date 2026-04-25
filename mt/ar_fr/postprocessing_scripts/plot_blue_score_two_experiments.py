"""
plot_training_comparison.py
Generates a dual-axis comparison plot (BLEU + eval_loss) from two NLLB training log files.

Usage:
    python plot_training_comparison.py <v1_log> <v2_log> [--output plot.png]

Example:
    python plot_training_comparison.py train_21388.log train_21395.log
"""

import re
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D


# ── parser ────────────────────────────────────────────────────────────────────

def parse_log(path: str) -> dict:
    """
    Extract baseline BLEU, and per-epoch eval_bleu / eval_loss from a log file.
    Returns:
        {
            "baseline_bleu": float | None,
            "epochs":        [float, ...],
            "bleu":          [float, ...],
            "loss":          [float, ...],
        }
    """
    baseline_bleu = None
    epochs, bleu, loss = [], [], []

    baseline_pattern = re.compile(r"eval_bleu['\"]?\s*:\s*([\d.]+)")
    eval_pattern     = re.compile(
        r"\{[^}]*'eval_bleu'\s*:\s*([\d.]+)[^}]*'eval_loss'\s*:\s*([\d.]+)[^}]*'epoch'\s*:\s*([\d.]+)[^}]*\}"
        r"|"
        r"\{[^}]*'eval_loss'\s*:\s*([\d.]+)[^}]*'eval_bleu'\s*:\s*([\d.]+)[^}]*'epoch'\s*:\s*([\d.]+)[^}]*\}"
    )
    baseline_section = False

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()

        # detect baseline block
        if "BASELINE" in line or "ZERO-SHOT" in line:
            baseline_section = True

        if baseline_section and "eval_bleu" in line and not epochs:
            m = baseline_pattern.search(line)
            if m:
                baseline_bleu = float(m.group(1))
                baseline_section = False
            continue

        # eval rows look like a Python dict on a single line
        if "'eval_bleu'" in line or '"eval_bleu"' in line:
            # flexible extraction — order of keys may vary
            b = re.search(r"'eval_bleu'\s*:\s*([\d.]+)", line)
            l = re.search(r"'eval_loss'\s*:\s*([\d.]+)", line)
            e = re.search(r"'epoch'\s*:\s*([\d.]+)", line)
            if b and l and e:
                epochs.append(float(e.group(1)))
                bleu.append(float(b.group(1)))
                loss.append(float(l.group(1)))

    return {"baseline_bleu": baseline_bleu, "epochs": epochs, "bleu": bleu, "loss": loss}


# ── plot ──────────────────────────────────────────────────────────────────────

def make_plot(v1: dict, v2: dict, output: str) -> None:

    RED   = "#E24B4A"
    GREEN = "#1D9E75"

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    def _epochs_with_baseline(data):
        """Prepend epoch 0 (baseline) if available."""
        e = data["epochs"][:]
        b = data["bleu"][:]
        if data["baseline_bleu"] is not None:
            e = [0.0] + e
            b = [data["baseline_bleu"]] + b
        return e, b

    # BLEU curves (left axis)
    e1, b1 = _epochs_with_baseline(v1)
    e2, b2 = _epochs_with_baseline(v2)

    ax1.plot(e1, b1, color=RED,   marker="o", ms=6, lw=2,   label="v1 BLEU (leak)")
    ax1.plot(e2, b2, color=GREEN, marker="o", ms=6, lw=2,   label="v2 BLEU (clean)")

    # eval_loss curves (right axis, dashed)
    ax2.plot(v1["epochs"], v1["loss"], color=RED,   marker="^", ms=5, lw=1.5,
             linestyle="--", label="v1 eval_loss")
    ax2.plot(v2["epochs"], v2["loss"], color=GREEN, marker="^", ms=5, lw=1.5,
             linestyle="--", label="v2 eval_loss")

    # axis labels & limits
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("BLEU", fontsize=12, color="black")
    ax2.set_ylabel("eval_loss", fontsize=12, color="grey")
    ax1.set_ylim(0, max(max(b1), max(b2)) * 1.15)
    ax2.set_ylim(
        min(min(v1["loss"]), min(v2["loss"])) * 0.995,
        max(max(v1["loss"]), max(v2["loss"])) * 1.005,
    )
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # grid on primary axis only
    ax1.grid(axis="y", color="grey", alpha=0.15, linestyle="--")
    ax1.set_axisbelow(True)

    # custom legend
    legend_elements = [
        Line2D([0], [0], color=RED,   marker="o", ms=6, lw=2,   label="v1 BLEU (data leak)"),
        Line2D([0], [0], color=GREEN, marker="o", ms=6, lw=2,   label="v2 BLEU (clean split)"),
        Line2D([0], [0], color=RED,   marker="^", ms=5, lw=1.5, linestyle="--", label="v1 eval_loss"),
        Line2D([0], [0], color=GREEN, marker="^", ms=5, lw=1.5, linestyle="--", label="v2 eval_loss"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=10, framealpha=0.9)

    ax1.set_title("NLLB fine-tuning: v1 (data leak) vs v2 (clean)\nArabic → French translation", fontsize=13)

    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {output}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot NLLB training logs.")
    parser.add_argument("v1_log", help="Path to v1 log file (with data leak)")
    parser.add_argument("v2_log", help="Path to v2 log file (clean split)")
    parser.add_argument("--output", default="training_comparison.png",
                        help="Output image path (default: training_comparison.png)")
    args = parser.parse_args()

    print(f"Parsing v1: {args.v1_log}")
    v1 = parse_log(args.v1_log)
    print(f"  baseline BLEU={v1['baseline_bleu']}  epochs={v1['epochs']}  BLEU={v1['bleu']}")

    print(f"Parsing v2: {args.v2_log}")
    v2 = parse_log(args.v2_log)
    print(f"  baseline BLEU={v2['baseline_bleu']}  epochs={v2['epochs']}  BLEU={v2['bleu']}")

    make_plot(v1, v2, args.output)


if __name__ == "__main__":
    main()

