"""
HERMES — Ablation Baseline
============================
Sends each tweet directly to Llama 4 Scout with a generic
sentiment classification prompt. No CPM pipeline, no SECs,
no explicit reasoning.

This establishes the zero-shot baseline to measure the
contribution of the CPM architecture.

Dataset: SemEval-2017 Task 4A
Classes: positive, negative, neutral
"""

import asyncio
import json
import os
from datetime import datetime

import openpyxl
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_PATH = "enrichment_agent/Datasets/SemEval_Dataset_Unido.xlsx"
OUTPUT_FILE = f"ablation_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
START_INDEX = 0
MAX_COMMENTS = 200

VALID_LABELS = {"positive", "negative", "neutral"}

PROMPT_TEMPLATE = """Classify the sentiment expressed in this tweet as exactly one of: positive, negative, or neutral.

Tweet: {tweet}

Respond with ONLY a JSON object in this exact format, nothing else:
{{"sentiment_label": "positive"}}
or
{{"sentiment_label": "negative"}}
or
{{"sentiment_label": "neutral"}}
"""


# ============================================================
# DATASET LOADER
# ============================================================

def load_comments(
    path: str,
    start: int = 0,
    limit: int | None = None,
) -> list[dict]:
    """Load tweets from the unified Excel dataset.

    Expected columns: tweet_id, gold_label, tweet_text.
    """
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb.active

    comments = []
    for i, row in enumerate(ws.iter_rows(min_row=2, values_only=True)):
        if i < start:
            continue

        tweet_id, gold_label, tweet_text = row
        gold_label = str(gold_label).strip().lower()

        if gold_label not in VALID_LABELS:
            print(f"Skipping row {i}: unknown label '{gold_label}'")
            continue

        comments.append({
            "index": i,
            "tweet_id": str(tweet_id),
            "gold_label": gold_label,
            "text": str(tweet_text).strip(),
        })

        if limit and len(comments) >= limit:
            break

    wb.close()
    return comments


# ============================================================
# MODEL
# ============================================================

def init_model():
    return ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0,
        max_tokens=256,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


# ============================================================
# RESPONSE PARSING
# ============================================================

LABEL_ALIASES = {
    "pos": "positive",
    "neg": "negative",
    "neu": "neutral",
    "neut": "neutral",
}


def parse_response(text: str) -> str:
    """Extract the sentiment_label from the model response."""
    text = text.strip()

    # Try JSON parsing
    try:
        clean = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        label = data.get("sentiment_label", "").strip().lower()
        if label in VALID_LABELS:
            return label
        mapped = LABEL_ALIASES.get(label)
        if mapped:
            return mapped
    except json.JSONDecodeError:
        pass

    # Fallback: find label in raw text
    text_lower = text.lower()
    for label in VALID_LABELS:
        if label in text_lower:
            return label

    # Fallback: aliases
    for alias, mapped in LABEL_ALIASES.items():
        if alias in text_lower:
            return mapped

    return ""


# ============================================================
# RUNNER
# ============================================================

async def run_baseline():
    comments = load_comments(DATASET_PATH, start=START_INDEX, limit=MAX_COMMENTS)

    dist = {l: sum(1 for c in comments if c["gold_label"] == l) for l in VALID_LABELS}
    print(f"Loaded {len(comments)} tweets (from index {START_INDEX})")
    print(f"  Distribution: {dist}")

    model = init_model()
    results = []

    for i, comment in enumerate(comments):
        print(f"\n[{i+1}/{len(comments)}] Processing: {comment['text'][:80]}...")

        try:
            prompt = PROMPT_TEMPLATE.format(tweet=comment["text"])
            response = await model.ainvoke([HumanMessage(content=prompt)])

            raw_response = response.content if isinstance(response.content, str) else ""
            prediction = parse_response(raw_response)

            if prediction:
                print(f"  -> {prediction}")
            else:
                print(f"  -> Could not parse: {raw_response[:100]}")

            result = {
                "index": comment["index"],
                "tweet_id": comment["tweet_id"],
                "comment": comment["text"],
                "gold_label": comment["gold_label"],
                "raw_response": raw_response,
                "convergence": {
                    "sentiment_label": prediction,
                    "reasoning": "Ablation baseline — zero-shot direct classification, no CPM pipeline.",
                },
            }

        except Exception as e:
            print(f"  Error: {e}")
            result = {
                "index": comment["index"],
                "tweet_id": comment["tweet_id"],
                "comment": comment["text"],
                "gold_label": comment["gold_label"],
                "error": str(e),
            }

        results.append(result)

        # Save progressively
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nAblation complete. Results saved to: {OUTPUT_FILE}")
    print(f"  Total: {len(results)} tweets processed")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(run_baseline())