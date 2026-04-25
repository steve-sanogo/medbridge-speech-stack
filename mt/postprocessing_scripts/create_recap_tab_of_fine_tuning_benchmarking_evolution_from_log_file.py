"""
Extract and display the evolution of BLEU evaluation scores (val & test)
from a training log file (ewe_train_21708.log format).

Usage:
    python extract_bleu_scores.py <log_file>
    python extract_bleu_scores.py  # defaults to ewe_train_21708.log
"""

import re
import sys
import json


def parse_log(filepath: str):
    val_scores = []   # list of (epoch, bleu)
    test_scores = []  # list of (epoch, bleu)  — initial (epoch 0) + final

    # Patterns
    re_initial = re.compile(
        r"INITIAL.*?Val BLEU:\s*([\d.]+).*?Test BLEU:\s*([\d.]+)", re.IGNORECASE
    )
    re_final = re.compile(
        r"FINAL.*?Val BLEU:\s*([\d.]+).*?Test BLEU:\s*([\d.]+)", re.IGNORECASE
    )
    re_eval = re.compile(
        r"\{.*?'eval_bleu':\s*([\d.]+).*?'epoch':\s*([\d.]+).*?\}"
    )

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            m = re_initial.search(line)
            if m:
                val_bleu, test_bleu = float(m.group(1)), float(m.group(2))
                val_scores.append((0.0, val_bleu))
                test_scores.append((0.0, test_bleu))
                continue

            m = re_eval.search(line)
            if m:
                bleu, epoch = float(m.group(1)), float(m.group(2))
                val_scores.append((epoch, bleu))
                continue

            m = re_final.search(line)
            if m:
                val_bleu, test_bleu = float(m.group(1)), float(m.group(2))
                # Final val is already captured from the last eval_bleu line,
                # but we add the test score at the last epoch
                last_epoch = val_scores[-1][0] if val_scores else 0.0
                test_scores.append((last_epoch, test_bleu))

    return val_scores, test_scores


def print_table(val_scores, test_scores):
    # Build a unified epoch list
    test_by_epoch = {ep: bleu for ep, bleu in test_scores}

    print(f"\n{'Epoch':>8}  {'Val BLEU':>10}  {'Test BLEU':>10}")
    print("-" * 34)
    for epoch, val_bleu in val_scores:
        test_bleu = test_by_epoch.get(epoch, None)
        test_str = f"{test_bleu:10.2f}" if test_bleu is not None else f"{'—':>10}"
        print(f"{epoch:8.1f}  {val_bleu:10.2f}  {test_str}")
    print()


def print_summary(val_scores, test_scores):
    if not val_scores:
        print("No evaluation scores found.")
        return

    best_val_epoch, best_val = max(val_scores, key=lambda x: x[1])
    print("=== Summary ===")
    print(f"  Initial Val  BLEU : {val_scores[0][1]:.2f}  (epoch {val_scores[0][0]})")
    print(f"  Final   Val  BLEU : {val_scores[-1][1]:.2f}  (epoch {val_scores[-1][0]})")
    print(f"  Best    Val  BLEU : {best_val:.2f}  (epoch {best_val_epoch})")

    if len(test_scores) >= 2:
        print(f"  Initial Test BLEU : {test_scores[0][1]:.2f}  (epoch {test_scores[0][0]})")
        print(f"  Final   Test BLEU : {test_scores[-1][1]:.2f}  (epoch {test_scores[-1][0]})")
    elif len(test_scores) == 1:
        print(f"  Test BLEU : {test_scores[0][1]:.2f}  (epoch {test_scores[0][0]})")
    print()


def export_json(val_scores, test_scores, out_path="bleu_scores.json"):
    data = {
        "val": [{"epoch": ep, "bleu": bleu} for ep, bleu in val_scores],
        "test": [{"epoch": ep, "bleu": bleu} for ep, bleu in test_scores],
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Scores exported to {out_path}")


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "ewe_train_21708.log"

    print(f"Parsing: {log_path}")
    val_scores, test_scores = parse_log(log_path)

    if not val_scores:
        print("No evaluation scores found. Check the log format.")
        sys.exit(1)

    print_table(val_scores, test_scores)
    print_summary(val_scores, test_scores)

    # Optionally export to JSON
    if "--json" in sys.argv:
        export_json(val_scores, test_scores)