"""
HERMES — Error Diagnostics & Benchmarks
==========================================
SemEval-2017 Task 4A — 3 classes: positive, negative, neutral

Metrics (in order of priority):
  AvgRec, Macro-F1, F1^PN, Accuracy, Micro-F1

Auto-detects ablation (sec_output) vs full pipeline (4 SECs) format.

Usage:
    python benchmarks.py [results.json] [output_dir]
"""

import json
import sys
import os
from collections import Counter

# ============================================================
# CONFIG
# ============================================================

LABELS = ["positive", "negative", "neutral"]
POLAR_LABELS = ["positive", "negative"]

LABEL_MAP = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
}


# ============================================================
# HELPERS
# ============================================================

def extract_pred(item):
    conv = item.get("convergence")
    if not conv or "sentiment_label" not in conv:
        return None
    return conv["sentiment_label"].lower().strip()


def extract_gold(item):
    gold = item.get("gold_label", "").lower().strip()
    return gold if gold in LABELS else None


def detect_format(item):
    """Detect whether the JSON is from ablation (sec_output) or full pipeline (4 SECs)."""
    if "sec_output" in item:
        return "ablation"
    if "relevance" in item:
        return "full"
    return "unknown"


def compute_metrics(items):
    """Compute all SemEval-2017 Task 4A metrics.

    Returns: stats, avg_rec, macro_f1, f1_pn, accuracy, micro_f1
    """
    stats = {}
    for cls in LABELS:
        tp = sum(1 for g, p in items if g == cls and p == cls)
        fp = sum(1 for g, p in items if g != cls and p == cls)
        fn = sum(1 for g, p in items if g == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        gold_n = sum(1 for g, _ in items if g == cls)
        pred_n = sum(1 for _, p in items if p == cls)
        stats[cls] = {
            "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
            "gold_count": gold_n, "pred_count": pred_n,
        }

    avg_rec = sum(stats[c]["recall"] for c in LABELS) / len(LABELS)
    macro_f1 = sum(stats[c]["f1"] for c in LABELS) / len(LABELS)
    f1_pn = sum(stats[c]["f1"] for c in POLAR_LABELS) / len(POLAR_LABELS)
    accuracy = sum(1 for g, p in items if g == p) / len(items) if items else 0

    total_tp = sum(stats[c]["tp"] for c in LABELS)
    total_fp = sum(stats[c]["fp"] for c in LABELS)
    total_fn = sum(stats[c]["fn"] for c in LABELS)
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) else 0

    return stats, avg_rec, macro_f1, f1_pn, accuracy, micro_f1


def key_to_filename(key):
    """Convert an error pattern key to a Windows-safe filename."""
    return key.replace(">", "_to_")


def format_trace(item):
    """Format reasoning trace — auto-detects ablation vs full format."""
    lines = []

    lines.append("### Tweet")
    lines.append("```")
    lines.append(item.get("comment", "[no text]"))
    lines.append("```")
    lines.append("")

    gold = extract_gold(item) or "?"
    lines.append(f"**Gold:** {gold}  ")

    pred = extract_pred(item) or "?"
    lines.append(f"**Predicted:** {pred}  ")
    loops = item.get("loop_counts", {})
    if loops:
        loop_str = ", ".join(f"{k}: {v}" for k, v in loops.items())
        lines.append(f"**Loops:** {loop_str}  ")
    lines.append("")

    fmt = detect_format(item)

    if fmt == "ablation":
        lines.append("#### SEC OUTPUT")
        lines.append("")
        sec = item.get("sec_output") or "[no output]"
        lines.append(sec.strip())
        lines.append("")

    else:
        lines.append("#### SEC 1 — RELEVANCE")
        lines.append("")
        lines.append((item.get("relevance") or "[no output]").strip())
        lines.append("")

        lines.append("#### SEC 2 — IMPLICATION")
        lines.append("")
        lines.append((item.get("implication") or "[no output]").strip())
        lines.append("")

        lines.append("#### SEC 3 — COPING")
        lines.append("")
        lines.append((item.get("coping") or "[no output]").strip())
        lines.append("")

        lines.append("#### SEC 4 — NORMATIVE")
        lines.append("")
        lines.append((item.get("normative") or "[no output]").strip())
        lines.append("")

    lines.append("#### CONVERGENCE")
    lines.append("")
    conv = item.get("convergence")
    if conv:
        lines.append(f"**Label:** {conv.get('sentiment_label', '?')}")
        lines.append("")
        lines.append(conv.get("reasoning", "[no reasoning]").strip())
    else:
        lines.append("[no convergence output]")
    lines.append("")

    return "\n".join(lines)


def write_metrics_table(lines, avg_rec, macro_f1, f1_pn, accuracy, micro_f1):
    """Append a metrics table to the summary lines."""
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| AvgRec | {avg_rec:.4f} |")
    lines.append(f"| Macro-F1 | {macro_f1:.4f} |")
    lines.append(f"| F1^PN | {f1_pn:.4f} |")
    lines.append(f"| Accuracy | {accuracy:.4f} |")
    lines.append(f"| Micro-F1 | {micro_f1:.4f} |")
    lines.append("")


# ============================================================
# AUTO-DETECT LATEST RESULTS FILE
# ============================================================

def find_latest_results():
    import glob
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(script_dir, "results_*.json")
    files = glob.glob(pattern)
    if not files:
        parent_dir = os.path.dirname(script_dir)
        pattern = os.path.join(parent_dir, "results_*.json")
        files = glob.glob(pattern)
    if not files:
        print("No results_*.json file found")
        sys.exit(1)
    files.sort()
    return files[-1]


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]
    else:
        input_path = find_latest_results()

    output_dir = sys.argv[2] if len(sys.argv) > 2 else "diagnostics_output"
    os.makedirs(output_dir, exist_ok=True)

    print(f"File: {os.path.basename(input_path)}")
    print(f"Output: {os.path.abspath(output_dir)}/")
    print()

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    # Detect format
    fmt = "unknown"
    if data:
        fmt = detect_format(data[0])
        print(f"Format detected: {fmt}")

    # Classify items
    evaluated = []
    skipped = 0
    for item in data:
        gold = extract_gold(item)
        pred = extract_pred(item)
        if not gold or not pred:
            skipped += 1
            continue
        if pred not in LABELS:
            skipped += 1
            continue
        item["_gold"] = gold
        item["_pred"] = pred
        item["_correct"] = gold == pred
        evaluated.append(item)

    # Compute metrics
    pairs = [(it["_gold"], it["_pred"]) for it in evaluated]
    stats, avg_rec, macro_f1, f1_pn, accuracy, micro_f1 = compute_metrics(pairs)

    # Group errors
    error_groups = {}
    correct_groups = {}
    for item in evaluated:
        if item["_correct"]:
            correct_groups.setdefault(item["_gold"], []).append(item)
        else:
            key = f"{item['_gold']}>{item['_pred']}"
            error_groups.setdefault(key, []).append(item)

    # ── Write summary.md ────────────────────────────────────
    summary_lines = []
    fmt_label = "Ablation (1 SEC)" if fmt == "ablation" else "Full Pipeline (4 SECs)"
    summary_lines.append(f"# HERMES — Error Diagnostics ({fmt_label})")
    summary_lines.append("")
    summary_lines.append(f"**File:** `{os.path.basename(input_path)}`  ")
    summary_lines.append(f"**Format:** {fmt_label}  ")
    summary_lines.append(f"**Classes:** positive, negative, neutral  ")
    summary_lines.append(f"**Total evaluated:** {len(evaluated)} (skipped: {skipped})  ")
    summary_lines.append(f"**Correct:** {sum(1 for it in evaluated if it['_correct'])}  ")
    summary_lines.append(f"**Errors:** {sum(1 for it in evaluated if not it['_correct'])}  ")
    summary_lines.append("")

    summary_lines.append("## Metrics")
    summary_lines.append("")
    write_metrics_table(summary_lines, avg_rec, macro_f1, f1_pn, accuracy, micro_f1)

    summary_lines.append("## Per-Class Metrics")
    summary_lines.append("")
    summary_lines.append("| Class | Precision | Recall | F1 | Gold | Pred |")
    summary_lines.append("|-------|-----------|--------|----|------|------|")
    for cls in LABELS:
        s = stats[cls]
        summary_lines.append(
            f"| {cls} | {s['precision']:.4f} | {s['recall']:.4f} | "
            f"{s['f1']:.4f} | {s['gold_count']} | {s['pred_count']} |"
        )
    summary_lines.append("")

    summary_lines.append("## Confusion Matrix")
    summary_lines.append("")
    header = "| Gold \\ Pred | " + " | ".join(LABEL_MAP[l] for l in LABELS) + " |"
    sep = "|------------|" + "|".join("-------:" for _ in LABELS) + "|"
    summary_lines.append(header)
    summary_lines.append(sep)
    confusion = Counter(pairs)
    for g in LABELS:
        row = [str(confusion.get((g, p), 0)) for p in LABELS]
        summary_lines.append(f"| {g} | {' | '.join(row)} |")
    summary_lines.append("")

    summary_lines.append("## Error Distribution")
    summary_lines.append("")
    summary_lines.append("| Pattern | Count | % of Total Errors | File |")
    summary_lines.append("|---------|-------|------------------:|------|")
    total_errors = sum(len(v) for v in error_groups.values())
    for key in sorted(error_groups.keys(), key=lambda k: -len(error_groups[k])):
        n = len(error_groups[key])
        fname = f"errors_{key_to_filename(key)}.md"
        pct = n / total_errors * 100 if total_errors else 0
        summary_lines.append(f"| {key} | {n} | {pct:.1f}% | `{fname}` |")
    summary_lines.append("")

    # Resource usage (tokens + time)
    total_input_tok = sum(it.get("input_tokens", 0) for it in evaluated)
    total_output_tok = sum(it.get("output_tokens", 0) for it in evaluated)
    total_elapsed = sum(it.get("elapsed_seconds", 0) for it in evaluated)
    n = len(evaluated)

    if n > 0 and (total_input_tok > 0 or total_elapsed > 0):
        avg_input_tok = total_input_tok / n
        avg_output_tok = total_output_tok / n
        avg_elapsed = total_elapsed / n

        summary_lines.append("## Resource Usage")
        summary_lines.append("")
        summary_lines.append("| Metric | Total | Avg per Tweet |")
        summary_lines.append("|--------|------:|--------------:|")
        summary_lines.append(f"| Input tokens | {total_input_tok:,} | {avg_input_tok:,.0f} |")
        summary_lines.append(f"| Output tokens | {total_output_tok:,} | {avg_output_tok:,.0f} |")
        summary_lines.append(f"| Total tokens | {total_input_tok + total_output_tok:,} | {avg_input_tok + avg_output_tok:,.0f} |")
        summary_lines.append(f"| Time (seconds) | {total_elapsed:,.1f} | {avg_elapsed:.1f} |")
        summary_lines.append("")

    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    # ── Write error files ───────────────────────────────────
    for key, items in error_groups.items():
        fname = f"errors_{key_to_filename(key)}.md"
        lines = []
        lines.append(f"# Errors: {key}")
        lines.append("")
        lines.append(f"**Total:** {len(items)} errors  ")
        lines.append(f"**Pattern:** Gold={key.split('>')[0]}, Predicted={key.split('>')[1]}  ")
        lines.append("")
        lines.append("---")
        lines.append("")
        for idx, item in enumerate(items, 1):
            lines.append(f"## Error {idx}/{len(items)} — ID: {item.get('index', '?')}")
            lines.append("")
            lines.append(format_trace(item))
            lines.append("---")
            lines.append("")
        with open(os.path.join(output_dir, fname), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # ── Write correct_all.md ────────────────────────────────
    lines = []
    lines.append("# Correct Classifications — All Cases")
    lines.append("")
    lines.append("---")
    lines.append("")
    for cls in LABELS:
        items = correct_groups.get(cls, [])
        lines.append(f"## Class: {cls} ({len(items)} correct)")
        lines.append("")
        for idx, item in enumerate(items, 1):
            lines.append(f"### Correct {idx} — ID: {item.get('index', '?')}")
            lines.append("")
            lines.append(format_trace(item))
            lines.append("---")
            lines.append("")
    with open(os.path.join(output_dir, "correct_all.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ── Print summary ───────────────────────────────────────
    print(f"\nDiagnostics generated in: {output_dir}/")
    print(f"  summary.md")
    for key in sorted(error_groups.keys(), key=lambda k: -len(error_groups[k])):
        fname = f"errors_{key_to_filename(key)}.md"
        print(f"  {fname} ({len(error_groups[key])} errors)")
    print(f"  correct_all.md ({sum(len(v) for v in correct_groups.values())} correct)")


if __name__ == "__main__":
    main()