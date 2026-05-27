"""
HERMES — Ablation Baseline
============================
Evalúa técnicas de clasificación sin pipeline CPM para comparar con Hermes.

CONFIGURACIÓN RÁPIDA:
  - TECHNIQUE: "zero_shot" o "cot"
  - DATASET: "tass2019PE"

El sistema de checkpoint permite retomar si el proceso se interrumpe.
"""

import asyncio
import json
import os
import re
import time
import xml.etree.ElementTree as ET

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()


# ============================================================
# CONFIGURACIÓN PRINCIPAL — CAMBIAR AQUÍ
# ============================================================

TECHNIQUE = "cot"   # "zero_shot" o "cot"
DATASET   = "tass2019PE"

MAX_COMMENTS = None

MAX_RETRIES = 3
BASE_DELAY = 2


# ============================================================
# CONFIGURACIÓN POR DATASET
# ============================================================

DATASET_CONFIG = {
    "tass2019PE": {
        "format": "xml",
        "path": "enrichment_agent/TASS2019_country_PE_train.xml",
        "valid_labels": {"P", "N", "NEU", "NONE"},
        "label_aliases": {
            "positivo":         "P",
            "negativo":         "N",
            "neutral":          "NEU",
            "ninguno":          "NONE",
            "sin sentimiento":  "NONE",
        },
        "task_description": (
            "Clasifica el sentimiento expresado en este tweet como exactamente uno de: "
            "P (positivo), N (negativo), NEU (neutral), NONE (sin sentimiento)."
        ),
        "output_format": (
            '{"sentiment_label": "P"}\n'
            'o\n'
            '{"sentiment_label": "N"}\n'
            'o\n'
            '{"sentiment_label": "NEU"}\n'
            'o\n'
            '{"sentiment_label": "NONE"}'
        ),
    },
}


# ============================================================
# PROMPTS POR TÉCNICA
# ============================================================

def build_prompt(technique: str, dataset: str, tweet: str) -> str:
    cfg  = DATASET_CONFIG[dataset]
    task = cfg["task_description"]
    fmt  = cfg["output_format"]

    if technique == "zero_shot":
        return f"""{task}

Tweet: {tweet}

Responde ÚNICAMENTE con un objeto JSON en este formato exacto, nada más:
{fmt}
"""

    elif technique == "cot":
        return f"""{task}

Tweet: {tweet}

Piensa paso a paso antes de responder:
1. ¿Cuál es el significado literal del tweet?
2. ¿Hay señales lingüísticas que indiquen que el tono del autor difiere del significado literal (p. ej. sarcasmo, ironía, exageración)?
3. ¿Cuál es la postura evaluativa real del autor hacia el tema?
4. Basándote en tu razonamiento, ¿cuál es la clasificación final?

Después de razonar, responde ÚNICAMENTE con un objeto JSON en este formato exacto, nada más:
{fmt}
"""

    else:
        raise ValueError(f"Técnica desconocida: '{technique}'. Usa 'zero_shot' o 'cot'.")


# ============================================================
# NOMBRE DEL ARCHIVO DE OUTPUT
# ============================================================

def get_output_filename(technique: str, dataset: str) -> str:
    return f"ablation_{technique}_{dataset}.json"


# ============================================================
# DATASET LOADER
# ============================================================

def load_comments_xml(dataset: str, limit: int | None = None) -> list[dict]:
    """Loader para TASS2019PE — archivo XML."""
    cfg          = DATASET_CONFIG[dataset]
    valid_labels = cfg["valid_labels"]

    tree = ET.parse(cfg["path"])
    root = tree.getroot()

    comments = []
    for i, tweet in enumerate(root.findall("tweet")):
        tweet_id   = tweet.findtext("tweetid", "").strip()
        text       = tweet.findtext("content", "").strip()
        gold_label = tweet.findtext("sentiment/polarity/value", "").strip().upper()

        if gold_label not in valid_labels:
            print(f"Skipping row {i}: unknown label '{gold_label}'")
            continue

        comments.append({
            "index":      i,
            "tweet_id":   tweet_id,
            "gold_label": gold_label,
            "text":       text,
        })

        if limit and len(comments) >= limit:
            break

    return comments


def load_comments(dataset: str, limit: int | None = None) -> list[dict]:
    fmt = DATASET_CONFIG[dataset]["format"]
    if fmt == "xml":
        return load_comments_xml(dataset, limit)
    else:
        raise ValueError(f"Unknown format: '{fmt}'")


# ============================================================
# MODELO
# ============================================================

def init_model() -> ChatGroq:
    return ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0,
        max_tokens=1024,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


# ============================================================
# PARSEO DE RESPUESTA
# ============================================================

def parse_response(text: str, dataset: str) -> str:
    cfg           = DATASET_CONFIG[dataset]
    valid_labels  = cfg["valid_labels"]
    label_aliases = cfg["label_aliases"]

    text = text.strip()

    # Intentar parseo JSON — tomar el último bloque JSON (útil para CoT)
    try:
        clean        = text.replace("```json", "").replace("```", "").strip()
        json_matches = re.findall(r'\{[^{}]+\}', clean)
        if json_matches:
            data  = json.loads(json_matches[-1])
            label = data.get("sentiment_label", "").strip().upper()
            if label in valid_labels:
                return label
            mapped = label_aliases.get(label.lower())
            if mapped and mapped in valid_labels:
                return mapped
    except (json.JSONDecodeError, Exception):
        pass

    # Fallback: buscar etiqueta válida en el texto
    text_upper = text.upper()
    for label in valid_labels:
        if label in text_upper:
            return label

    # Fallback: aliases
    text_lower = text.lower()
    for alias, mapped in label_aliases.items():
        if alias in text_lower and mapped in valid_labels:
            return mapped

    return ""


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint(output_file: str) -> tuple[list[dict], set[int]]:
    if not os.path.exists(output_file):
        return [], set()

    with open(output_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    processed = {r["index"] for r in results}
    print(f"Checkpoint cargado: {len(results)} tweets ya procesados")
    return results, processed


# ============================================================
# RETRY
# ============================================================

def _is_retryable(error: Exception) -> bool:
    error_str = str(error).lower()
    signals   = ["429", "rate limit", "500", "502", "503", "504",
                 "server error", "timeout", "timed out", "overloaded"]
    return any(s in error_str for s in signals)


async def invoke_with_retry(model: ChatGroq, prompt: str) -> str:
    msg = [HumanMessage(content=prompt)]
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = await model.ainvoke(msg)
            return response.content if isinstance(response.content, str) else ""
        except Exception as e:
            if attempt < MAX_RETRIES and _is_retryable(e):
                delay = BASE_DELAY * (2 ** attempt)
                print(f"  Retryable error (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                print(f"  Waiting {delay}s...")
                await asyncio.sleep(delay)
            else:
                raise


# ============================================================
# RUNNER PRINCIPAL
# ============================================================

async def run_ablation():
    output_file = get_output_filename(TECHNIQUE, DATASET)

    print(f"\n{'='*60}")
    print(f"  HERMES Ablation Baseline")
    print(f"  Técnica  : {TECHNIQUE.upper()}")
    print(f"  Dataset  : {DATASET}")
    print(f"  Output   : {output_file}")
    print(f"{'='*60}\n")

    comments = load_comments(DATASET, limit=MAX_COMMENTS)
    cfg       = DATASET_CONFIG[DATASET]
    dist      = {l: sum(1 for c in comments if c["gold_label"] == l) for l in cfg["valid_labels"]}
    print(f"Cargados {len(comments)} tweets")
    print(f"  Distribución: {dist}")

    results, processed_indices = load_checkpoint(output_file)
    remaining = [c for c in comments if c["index"] not in processed_indices]

    if not remaining:
        print("\nTodos los tweets ya fueron procesados.")
        return

    if processed_indices:
        print(f"Retomando: {len(remaining)} tweets pendientes")

    model       = init_model()
    batch_start = time.time()

    for i, comment in enumerate(remaining):
        progress = len(processed_indices) + i + 1
        total    = len(comments)
        print(f"\n[{progress}/{total}] {comment['text'][:80]}...")

        tweet_start = time.time()

        try:
            prompt       = build_prompt(TECHNIQUE, DATASET, comment["text"])
            raw_response = await invoke_with_retry(model, prompt)
            elapsed      = time.time() - tweet_start
            prediction   = parse_response(raw_response, DATASET)

            if prediction:
                print(f"  -> {prediction}  ({elapsed:.1f}s)")
            else:
                print(f"  -> No se pudo parsear: {raw_response[:100]}  ({elapsed:.1f}s)")

            result = {
                "index":      comment["index"],
                "tweet_id":   comment["tweet_id"],
                "comment":    comment["text"],
                "gold_label": comment["gold_label"],
                "raw_response": raw_response,
                "convergence": {
                    "sentiment_label": prediction,
                    "reasoning": f"Ablation {TECHNIQUE} — sin pipeline CPM.",
                },
                "technique":       TECHNIQUE,
                "dataset":         DATASET,
                "elapsed_seconds": round(elapsed, 2),
            }

        except Exception as e:
            elapsed = time.time() - tweet_start
            print(f"  Error ({elapsed:.1f}s): {e}")
            result = {
                "index":      comment["index"],
                "tweet_id":   comment["tweet_id"],
                "comment":    comment["text"],
                "gold_label": comment["gold_label"],
                "error":      str(e),
                "technique":  TECHNIQUE,
                "dataset":    DATASET,
                "elapsed_seconds": round(elapsed, 2),
            }

        results.append(result)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    batch_elapsed = time.time() - batch_start
    print(f"\nAblación completa. Resultados en: {output_file}")
    print(f"  Procesados esta corrida : {len(remaining)} tweets en {batch_elapsed:.1f}s")
    print(f"  Total en archivo        : {len(results)} tweets")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(run_ablation())