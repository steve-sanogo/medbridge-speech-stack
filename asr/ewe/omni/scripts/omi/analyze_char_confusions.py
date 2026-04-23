import argparse
from collections import Counter, defaultdict
import pandas as pd


def levenshtein_char_ops(ref: str, hyp: str):
    """Retourne la séquence d'opérations caractère à caractère."""
    r = list(ref)
    h = list(hyp)

    n = len(r)
    m = len(h)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution / match
            )

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if r[i - 1] == h[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost == 0:
                    ops.append(("match", r[i - 1], h[j - 1]))
                else:
                    ops.append(("sub", r[i - 1], h[j - 1]))
                i -= 1
                j -= 1
                continue

        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", r[i - 1], "∅"))
            i -= 1
            continue

        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("ins", "∅", h[j - 1]))
            j -= 1
            continue

    ops.reverse()
    return ops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to eval_results.csv")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    sub_counter = Counter()
    del_counter = Counter()
    ins_counter = Counter()
    char_counter_ref = Counter()
    char_counter_hyp = Counter()

    confusion_examples = defaultdict(list)

    target_chars = set(["ɛ", "ɔ", "ɖ", "ʋ", "ɣ", "Ɖ", "Ɛ", "Ɔ", "Ʋ", "Ɣ"])

    for _, row in df.iterrows():
        ref = str(row["reference"])
        hyp = str(row["hypothesis"])

        char_counter_ref.update(ref)
        char_counter_hyp.update(hyp)

        ops = levenshtein_char_ops(ref, hyp)

        for op, a, b in ops:
            if op == "sub":
                sub_counter[(a, b)] += 1
                if len(confusion_examples[(a, b)]) < args.examples:
                    confusion_examples[(a, b)].append((ref, hyp))
            elif op == "del":
                del_counter[a] += 1
            elif op == "ins":
                ins_counter[b] += 1

    print("\n" + "=" * 70)
    print("TOP SUBSTITUTIONS")
    print("=" * 70)
    for (a, b), c in sub_counter.most_common(args.top_k):
        print(f"{a!r} -> {b!r} : {c}")

    print("\n" + "=" * 70)
    print("TOP DELETIONS")
    print("=" * 70)
    for a, c in del_counter.most_common(args.top_k):
        print(f"{a!r} deleted : {c}")

    print("\n" + "=" * 70)
    print("TOP INSERTIONS")
    print("=" * 70)
    for b, c in ins_counter.most_common(args.top_k):
        print(f"{b!r} inserted : {c}")

    print("\n" + "=" * 70)
    print("FOCUS ON EWE-SPECIFIC CHARACTERS")
    print("=" * 70)
    for (a, b), c in sub_counter.most_common():
        if a in target_chars or b in target_chars or (a, b) in {
            ("ɛ", "e"), ("e", "ɛ"),
            ("ɔ", "o"), ("o", "ɔ"),
            ("ɖ", "d"), ("d", "ɖ"),
            ("ʋ", "v"), ("v", "ʋ"),
            ("ɣ", "g"), ("g", "ɣ"),
        }:
            print(f"{a!r} -> {b!r} : {c}")

    print("\n" + "=" * 70)
    print("CHARACTER FREQUENCIES (REFERENCE)")
    print("=" * 70)
    for ch in ["ɛ", "ɔ", "ɖ", "ʋ", "ɣ", "e", "o", "d", "v", "g"]:
        print(f"{ch!r} : {char_counter_ref[ch]}")

    print("\n" + "=" * 70)
    print("CHARACTER FREQUENCIES (HYPOTHESIS)")
    print("=" * 70)
    for ch in ["ɛ", "ɔ", "ɖ", "ʋ", "ɣ", "e", "o", "d", "v", "g"]:
        print(f"{ch!r} : {char_counter_hyp[ch]}")

    print("\n" + "=" * 70)
    print("EXAMPLES FOR KEY CONFUSIONS")
    print("=" * 70)

    key_pairs = [
        ("ɛ", "e"), ("e", "ɛ"),
        ("ɔ", "o"), ("o", "ɔ"),
        ("ɖ", "d"), ("d", "ɖ"),
        ("ʋ", "v"), ("v", "ʋ"),
        ("ɣ", "g"), ("g", "ɣ"),
    ]

    for pair in key_pairs:
        examples = confusion_examples.get(pair, [])
        if examples:
            print(f"\n### {pair[0]!r} -> {pair[1]!r}")
            for ref, hyp in examples[:args.examples]:
                print(f"REF: {ref}")
                print(f"HYP: {hyp}")
                print("-" * 40)


if __name__ == "__main__":
    main()