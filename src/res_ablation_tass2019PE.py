"""
HERMES — Ablation Benchmarks (TASS 2019 Perú)
===============================================
Evalúa y compara las técnicas de ablación (zero_shot, cot) contra
el pipeline completo (si está disponible), usando las métricas
oficiales de TASS 2019.

Métricas (en orden de prioridad):
  AvgRec, Macro-F1, F1^PN, Accuracy, Micro-F1

Formatos de entrada soportados:
  - ablation_zero_shot_tass2019PE.json  (technique: "zero_shot")
  - ablation_cot_tass2019PE.json        (technique: "cot")
  - results_tass2019PE*.json            (pipeline completo)

Uso:
    # Comparar zero_shot y cot automáticamente:
    python res-ablation-tass2019PE.py

    # Especificar archivos manualmente:
    python res-ablation-tass2019PE.py \\
        ablation_zero_shot_tass2019PE.json \\
        ablation_cot_tass2019PE.json \\
        [results_full.json] \\
        [--output diagnostics_ablation_tass2019PE]

    # Un solo archivo:
    python res-ablation-tass2019PE.py ablation_cot_tass2019PE.json
"""

import json
import sys
import os
import glob
import argparse
from collections import Counter

# ============================================================
# CONFIG
# ============================================================

LABELS       = ["P", "N", "NEU", "NONE"]
POLAR_LABELS = ["P", "N"]

LABEL_MAP = {
    "P":    "Positivo",
    "N":    "Negativo",
    "NEU":  "Neutral",
    "NONE": "Sin sentimiento",
}

TECHNIQUE_DISPLAY = {
    "zero_shot": "Zero-Shot",
    "cot":       "Chain-of-Thought (CoT)",
    "full":      "Full Pipeline (4 SECs)",
}


# ============================================================
# HELPERS — EXTRACTION
# ============================================================

def extract_pred(item: dict) -> str | None:
    conv = item.get("convergence")
    if not conv or "sentiment_label" not in conv:
        return None
    return conv["sentiment_label"].strip().upper()


def extract_gold(item: dict) -> str | None:
    gold = item.get("gold_label", "").strip().upper()
    return gold if gold in LABELS else None


def detect_technique(item: dict) -> str:
    technique = item.get("technique", "")
    if technique in ("zero_shot", "cot"):
        return technique
    if "relevance" in item:
        return "full"
    return "unknown"


# ============================================================
# HELPERS — MÉTRICAS
# ============================================================

def compute_metrics(pairs: list[tuple[str, str]]) -> dict:
    stats = {}
    for cls in LABELS:
        tp = sum(1 for g, p in pairs if g == cls and p == cls)
        fp = sum(1 for g, p in pairs if g != cls and p == cls)
        fn = sum(1 for g, p in pairs if g == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        stats[cls] = {
            "precision":  prec,
            "recall":     rec,
            "f1":         f1,
            "tp":         tp,
            "fp":         fp,
            "fn":         fn,
            "gold_count": sum(1 for g, _ in pairs if g == cls),
            "pred_count": sum(1 for _, p in pairs if p == cls),
        }

    avg_rec  = sum(stats[c]["recall"] for c in LABELS) / len(LABELS)
    macro_f1 = sum(stats[c]["f1"] for c in LABELS) / len(LABELS)
    f1_pn    = sum(stats[c]["f1"] for c in POLAR_LABELS) / len(POLAR_LABELS)
    accuracy = sum(1 for g, p in pairs if g == p) / len(pairs) if pairs else 0.0

    total_tp = sum(stats[c]["tp"] for c in LABELS)
    total_fp = sum(stats[c]["fp"] for c in LABELS)
    total_fn = sum(stats[c]["fn"] for c in LABELS)
    micro_p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    return {
        "stats":    stats,
        "avg_rec":  avg_rec,
        "macro_f1": macro_f1,
        "f1_pn":    f1_pn,
        "accuracy": accuracy,
        "micro_f1": micro_f1,
    }


# ============================================================
# HELPERS — CARGA Y CLASIFICACIÓN DE ÍTEMS
# ============================================================

def load_and_classify(path: str) -> tuple[list[dict], int]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    evaluated, skipped = [], 0
    for item in data:
        gold = extract_gold(item)
        pred = extract_pred(item)
        if not gold or not pred or pred not in LABELS:
            skipped += 1
            continue
        item["_gold"]      = gold
        item["_pred"]      = pred
        item["_correct"]   = gold == pred
        item["_technique"] = detect_technique(item)
        evaluated.append(item)

    return evaluated, skipped


# ============================================================
# HELPERS — FORMATO DE TRAZA
# ============================================================

def format_trace(item: dict) -> str:
    lines = []
    lines.append("### Tweet")
    lines.append("```")
    lines.append(item.get("comment", "[sin texto]"))
    lines.append("```")
    lines.append("")

    gold      = item.get("_gold") or extract_gold(item) or "?"
    pred      = item.get("_pred") or extract_pred(item) or "?"
    technique = item.get("_technique", item.get("technique", "?"))

    lines.append(f"**Gold:** {gold}  ")
    lines.append(f"**Predicted:** {pred}  ")
    lines.append(f"**Technique:** {TECHNIQUE_DISPLAY.get(technique, technique)}  ")
    lines.append(f"**Elapsed:** {item.get('elapsed_seconds', '?')}s  ")
    lines.append("")

    if technique == "zero_shot":
        lines.append("#### RAW RESPONSE (Zero-Shot)")
        lines.append("")
        lines.append((item.get("raw_response") or "[sin respuesta]").strip())
        lines.append("")

    elif technique == "cot":
        lines.append("#### CHAIN-OF-THOUGHT REASONING")
        lines.append("")
        lines.append((item.get("raw_response") or "[sin respuesta]").strip())
        lines.append("")

    elif technique == "full":
        for sec_key, sec_label in [
            ("relevance",   "SEC 1 — RELEVANCIA"),
            ("implication", "SEC 2 — IMPLICACIÓN"),
            ("coping",      "SEC 3 — AFRONTAMIENTO"),
            ("normative",   "SEC 4 — NORMATIVO"),
        ]:
            lines.append(f"#### {sec_label}")
            lines.append("")
            lines.append((item.get(sec_key) or "[sin output]").strip())
            lines.append("")

    lines.append("#### CONVERGENCIA")
    lines.append("")
    conv = item.get("convergence")
    if conv:
        lines.append(f"**Label:** {conv.get('sentiment_label', '?')}")
        lines.append("")
        lines.append(conv.get("reasoning", "[sin razonamiento]").strip())
    else:
        lines.append("[sin convergence output]")
    lines.append("")

    return "\n".join(lines)


# ============================================================
# HELPERS — ESCRITURA MARKDOWN
# ============================================================

def write_metrics_table(lines: list[str], m: dict) -> None:
    lines += [
        "| Métrica | Valor |",
        "|---------|-------|",
        f"| AvgRec    | {m['avg_rec']:.4f} |",
        f"| Macro-F1  | {m['macro_f1']:.4f} |",
        f"| F1^PN     | {m['f1_pn']:.4f} |",
        f"| Accuracy  | {m['accuracy']:.4f} |",
        f"| Micro-F1  | {m['micro_f1']:.4f} |",
        "",
    ]


def write_per_class_table(lines: list[str], stats: dict) -> None:
    lines += [
        "| Clase | Precision | Recall | F1 | Gold | Pred |",
        "|-------|-----------|--------|----|------|------|",
    ]
    for cls in LABELS:
        s = stats[cls]
        lines.append(
            f"| {cls} | {s['precision']:.4f} | {s['recall']:.4f} | "
            f"{s['f1']:.4f} | {s['gold_count']} | {s['pred_count']} |"
        )
    lines.append("")


def write_confusion_matrix(lines: list[str], pairs: list[tuple]) -> None:
    header = "| Gold \\ Pred | " + " | ".join(LABEL_MAP[l] for l in LABELS) + " |"
    sep    = "|------------|" + "|".join("-------:" for _ in LABELS) + "|"
    lines += [header, sep]
    confusion = Counter(pairs)
    for g in LABELS:
        row = [str(confusion.get((g, p), 0)) for p in LABELS]
        lines.append(f"| {g} | {' | '.join(row)} |")
    lines.append("")


def key_to_filename(key: str) -> str:
    return key.replace(">", "_to_")


# ============================================================
# HELPERS — DETECCIÓN AUTOMÁTICA DE ARCHIVOS
# ============================================================

def find_ablation_files() -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    for pattern in [
        "ablation_*_tass2019PE.json",
        "ablation_tass2019PE*.json",
    ]:
        candidates.extend(glob.glob(os.path.join(script_dir, pattern)))

    if not candidates:
        for pattern in [
            "ablation_*_tass2019PE.json",
            "ablation_tass2019PE*.json",
        ]:
            candidates.extend(glob.glob(pattern))

    return sorted(set(candidates))


def find_full_pipeline_file() -> str | None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for pattern in ["results_tass2019PE*.json"]:
        files = glob.glob(os.path.join(script_dir, pattern))
        if files:
            return sorted(files)[-1]
    return None


# ============================================================
# GENERACIÓN DE REPORTE POR TÉCNICA
# ============================================================

def generate_report(
    evaluated: list[dict],
    skipped: int,
    technique: str,
    source_file: str,
    output_dir: str,
) -> dict:
    tech_label = TECHNIQUE_DISPLAY.get(technique, technique)
    tech_dir   = os.path.join(output_dir, technique)
    os.makedirs(tech_dir, exist_ok=True)

    pairs = [(it["_gold"], it["_pred"]) for it in evaluated]
    m     = compute_metrics(pairs)

    error_groups   = {}
    correct_groups = {}
    for item in evaluated:
        if item["_correct"]:
            correct_groups.setdefault(item["_gold"], []).append(item)
        else:
            key = f"{item['_gold']}>{item['_pred']}"
            error_groups.setdefault(key, []).append(item)

    total_errors  = sum(len(v) for v in error_groups.values())
    total_correct = sum(len(v) for v in correct_groups.values())
    total_elapsed = sum(it.get("elapsed_seconds", 0) for it in evaluated)
    n             = len(evaluated)

    lines = [
        f"# HERMES — Ablation Benchmarks ({tech_label})",
        "",
        f"**Archivo fuente:** `{os.path.basename(source_file)}`  ",
        f"**Técnica:** {tech_label}  ",
        f"**Dataset:** TASS 2019 Perú  ",
        f"**Clases:** P, N, NEU, NONE  ",
        f"**Total evaluados:** {n} (omitidos: {skipped})  ",
        f"**Correctos:** {total_correct}  ",
        f"**Errores:** {total_errors}  ",
        "",
        "## Métricas",
        "",
    ]
    write_metrics_table(lines, m)

    lines += ["## Métricas por Clase", ""]
    write_per_class_table(lines, m["stats"])

    lines += ["## Matriz de Confusión", ""]
    write_confusion_matrix(lines, pairs)

    lines += ["## Distribución de Errores", "",
              "| Patrón | Count | % de Errores | Archivo |",
              "|--------|-------|-------------:|---------|"]
    for key in sorted(error_groups, key=lambda k: -len(error_groups[k])):
        n_err = len(error_groups[key])
        fname = f"errors_{key_to_filename(key)}.md"
        pct   = n_err / total_errors * 100 if total_errors else 0
        lines.append(f"| {key} | {n_err} | {pct:.1f}% | `{fname}` |")
    lines.append("")

    if n > 0 and total_elapsed > 0:
        lines += [
            "## Uso de Recursos",
            "",
            "| Métrica | Total | Avg por Tweet |",
            "|---------|------:|--------------:|",
            f"| Tiempo (segundos) | {total_elapsed:,.1f} | {total_elapsed/n:.2f} |",
            "",
        ]

    with open(os.path.join(tech_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    for key, items in error_groups.items():
        fname    = f"errors_{key_to_filename(key)}.md"
        g_label, p_label = key.split(">")
        err_lines = [
            f"# Errores: {key}",
            "",
            f"**Total:** {len(items)} errores  ",
            f"**Patrón:** Gold={g_label}, Predicted={p_label}  ",
            "",
            "---",
            "",
        ]
        for idx, item in enumerate(items, 1):
            err_lines += [
                f"## Error {idx}/{len(items)} — ID: {item.get('index', '?')}",
                "",
                format_trace(item),
                "---",
                "",
            ]
        with open(os.path.join(tech_dir, fname), "w", encoding="utf-8") as f:
            f.write("\n".join(err_lines))

    corr_lines = [
        "# Clasificaciones Correctas — Todos los Casos",
        "",
        "---",
        "",
    ]
    for cls in LABELS:
        items = correct_groups.get(cls, [])
        corr_lines += [f"## Clase: {cls} ({len(items)} correctos)", ""]
        for idx, item in enumerate(items, 1):
            corr_lines += [
                f"### Correcto {idx} — ID: {item.get('index', '?')}",
                "",
                format_trace(item),
                "---",
                "",
            ]
    with open(os.path.join(tech_dir, "correct_all.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(corr_lines))

    print(f"\n  [{tech_label}] → {tech_dir}/")
    print(f"    summary.md")
    for key in sorted(error_groups, key=lambda k: -len(error_groups[k])):
        fname = f"errors_{key_to_filename(key)}.md"
        print(f"    {fname} ({len(error_groups[key])} errores)")
    print(f"    correct_all.md ({total_correct} correctos)")

    return {"technique": technique, "label": tech_label, "metrics": m, "n": n}


# ============================================================
# REPORTE COMPARATIVO
# ============================================================

def generate_comparison(reports: list[dict], output_dir: str) -> None:
    if len(reports) < 2:
        return

    lines = [
        "# HERMES — Comparativa de Ablación (TASS 2019 Perú)",
        "",
        "## Métricas Principales",
        "",
        "| Técnica | N | AvgRec | Macro-F1 | F1^PN | Accuracy | Micro-F1 |",
        "|---------|---|--------|----------|-------|----------|----------|",
    ]
    for r in reports:
        m = r["metrics"]
        lines.append(
            f"| {r['label']} | {r['n']} "
            f"| {m['avg_rec']:.4f} | {m['macro_f1']:.4f} "
            f"| {m['f1_pn']:.4f} | {m['accuracy']:.4f} "
            f"| {m['micro_f1']:.4f} |"
        )
    lines.append("")

    lines += ["## Mejor Técnica por Métrica", ""]
    for metric_key, metric_name in [
        ("avg_rec",  "AvgRec"),
        ("macro_f1", "Macro-F1"),
        ("f1_pn",    "F1^PN"),
        ("accuracy", "Accuracy"),
        ("micro_f1", "Micro-F1"),
    ]:
        best = max(reports, key=lambda r: r["metrics"][metric_key])
        lines.append(f"- **{metric_name}**: {best['label']} ({best['metrics'][metric_key]:.4f})")
    lines.append("")

    lines += ["## Métricas por Clase (por Técnica)", ""]
    for cls in LABELS:
        lines += [f"### {LABEL_MAP[cls]}", "",
                  "| Técnica | Precision | Recall | F1 | Gold | Pred |",
                  "|---------|-----------|--------|----|------|------|"]
        for r in reports:
            s = r["metrics"]["stats"][cls]
            lines.append(
                f"| {r['label']} | {s['precision']:.4f} | {s['recall']:.4f} "
                f"| {s['f1']:.4f} | {s['gold_count']} | {s['pred_count']} |"
            )
        lines.append("")

    out_path = os.path.join(output_dir, "comparison.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Comparativa → {out_path}")


# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HERMES — Ablation Benchmarks TASS 2019 Perú"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Archivos JSON de ablación (si no se especifican, se detectan automáticamente)",
    )
    parser.add_argument(
        "--output", "-o",
        default="diagnostics_ablation_tass2019PE",
        help="Directorio de salida (default: diagnostics_ablation_tass2019PE)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.files:
        input_files = args.files
    else:
        input_files = find_ablation_files()
        full_file   = find_full_pipeline_file()
        if full_file and full_file not in input_files:
            input_files.append(full_file)

    if not input_files:
        print("ERROR: No se encontraron archivos de ablación.")
        print("  Uso: python res-ablation-tass2019PE.py <archivo1.json> [archivo2.json] ...")
        sys.exit(1)

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output: {os.path.abspath(output_dir)}/")
    print(f"Archivos a evaluar: {len(input_files)}")
    for f in input_files:
        print(f"  - {os.path.basename(f)}")

    all_reports = []

    for path in input_files:
        if not os.path.exists(path):
            print(f"\nWARNING: Archivo no encontrado: {path} — omitiendo.")
            continue

        print(f"\nCargando: {os.path.basename(path)}")
        evaluated, skipped = load_and_classify(path)

        if not evaluated:
            print(f"  WARNING: Sin ítems evaluables en {path}")
            continue

        technique = evaluated[0].get("_technique", "unknown")
        print(f"  Técnica detectada: {TECHNIQUE_DISPLAY.get(technique, technique)}")
        print(f"  Evaluados: {len(evaluated)} | Omitidos: {skipped}")

        report = generate_report(
            evaluated=evaluated,
            skipped=skipped,
            technique=technique,
            source_file=path,
            output_dir=output_dir,
        )
        all_reports.append(report)

    if len(all_reports) >= 2:
        generate_comparison(all_reports, output_dir)

    print("\n" + "=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    header = f"{'Técnica':<30} {'AvgRec':>8} {'MacroF1':>8} {'F1^PN':>8} {'Acc':>8}"
    print(header)
    print("-" * 60)
    for r in all_reports:
        m = r["metrics"]
        print(
            f"{r['label']:<30} {m['avg_rec']:>8.4f} {m['macro_f1']:>8.4f} "
            f"{m['f1_pn']:>8.4f} {m['accuracy']:>8.4f}"
        )
    print("=" * 60)
    print(f"\nDiagnósticos guardados en: {os.path.abspath(output_dir)}/")


if __name__ == "__main__":
    main()