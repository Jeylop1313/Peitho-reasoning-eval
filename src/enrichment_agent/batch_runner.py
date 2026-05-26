"""
HERMES — Batch Runner
========================
Loads tweets from TASS2019_country_PE_train.xml (TASS 2019 Perú format),
runs the full pipeline (4 SECs + Convergence) on each tweet,
and saves results progressively to JSON.

Supports checkpointing: if interrupted, re-running the same command
resumes from where it left off.

Includes retry with exponential backoff for rate limits (429) and
server errors (500).
"""

import asyncio
import json
import os
import time
import xml.etree.ElementTree as ET

from dotenv import load_dotenv
from enrichment_agent.Graph import graph

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_PATH = "enrichment_agent/Datasets/TASS2019_country_PE_train.xml"
OUTPUT_FILE = "results_tass2019PE.json"
AVATAR = "Usuario hispanohablante de Twitter, Perú"
START_INDEX = 0
MAX_COMMENTS = None

VALID_LABELS = {"P", "N", "NEU", "NONE"}

MAX_RETRIES = 3
BASE_DELAY = 2  # seconds, doubles each retry

CONVERGENCE_CONFIG = {
    "configurable": {
        "convergence_variant": "tass2019PE"
    }
}


# ============================================================
# DATASET LOADER
# ============================================================

def load_comments(
    path: str,
    start: int = 0,
    limit: int | None = None,
) -> list[dict]:
    """Load tweets from TASS2019 XML dataset.

    Expected fields: tweetid, content, sentiment/polarity/value.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    comments = []
    for i, tweet in enumerate(root.findall("tweet")):
        if i < start:
            continue

        tweet_id = tweet.findtext("tweetid", "").strip()
        text = tweet.findtext("content", "").strip()
        gold_label = tweet.findtext("sentiment/polarity/value", "").strip().upper()

        if gold_label not in VALID_LABELS:
            print(f"Skipping row {i}: unknown label '{gold_label}'")
            continue

        comments.append({
            "index": i,
            "tweet_id": tweet_id,
            "gold_label": gold_label,
            "text": text,
        })

        if limit and len(comments) >= limit:
            break

    return comments


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def load_checkpoint(path: str) -> tuple[list[dict], set[int]]:
    """Load existing results and extract processed indices."""
    if not os.path.exists(path):
        return [], set()

    with open(path, "r", encoding="utf-8") as f:
        results = json.load(f)

    processed = {r["index"] for r in results}
    print(f"Checkpoint loaded: {len(results)} tweets already processed")
    return results, processed


# ============================================================
# RETRY HELPERS
# ============================================================

def _is_retryable(error: Exception) -> bool:
    """Check if an error is retryable (rate limit or server error)."""
    error_str = str(error).lower()
    retryable_signals = ["429", "rate limit", "500", "502", "503", "504",
                         "server error", "timeout", "timed out", "overloaded"]
    return any(signal in error_str for signal in retryable_signals)


async def invoke_with_retry(topic: str, avatar: str) -> dict:
    """Call graph.ainvoke with exponential backoff on retryable errors."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await graph.ainvoke(
                {"topic": topic, "perfil_avatar": avatar},
                config=CONVERGENCE_CONFIG,
            )
        except Exception as e:
            if attempt < MAX_RETRIES and _is_retryable(e):
                delay = BASE_DELAY * (2 ** attempt)
                print(f"  Retryable error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                print(f"  Waiting {delay}s before retry...")
                await asyncio.sleep(delay)
            else:
                raise


# ============================================================
# RUNNER
# ============================================================

async def run_batch():
    comments = load_comments(DATASET_PATH, start=START_INDEX, limit=MAX_COMMENTS)

    dist = {l: sum(1 for c in comments if c["gold_label"] == l) for l in VALID_LABELS}
    print(f"Loaded {len(comments)} tweets (from index {START_INDEX})")
    print(f"  Distribution: {dist}")

    # Load checkpoint
    results, processed_indices = load_checkpoint(OUTPUT_FILE)
    remaining = [c for c in comments if c["index"] not in processed_indices]

    if not remaining:
        print("All tweets already processed. Nothing to do.")
        return

    if processed_indices:
        print(f"Resuming: {len(remaining)} tweets remaining")

    batch_start = time.time()
    total_input_tokens = 0
    total_output_tokens = 0

    for i, comment in enumerate(remaining):
        progress = len(processed_indices) + i + 1
        total = len(comments)
        print(f"\n[{progress}/{total}] Processing: {comment['text'][:80]}...")

        tweet_start = time.time()

        try:
            state = await invoke_with_retry(comment["text"], AVATAR)

            elapsed = time.time() - tweet_start
            input_tok = state.get("total_input_tokens", 0)
            output_tok = state.get("total_output_tokens", 0)
            total_input_tokens += input_tok
            total_output_tokens += output_tok

            print(f"  {elapsed:.1f}s | tokens: {input_tok} in / {output_tok} out")

            result = {
                "index": comment["index"],
                "tweet_id": comment["tweet_id"],
                "comment": comment["text"],
                "gold_label": comment["gold_label"],
                "relevance": state.get("relevance"),
                "implication": state.get("implication"),
                "coping": state.get("coping"),
                "normative": state.get("normative"),
                "convergence": state.get("convergence"),
                "loop_counts": {
                    "relevance": state.get("relevance_loop", 0),
                    "implication": state.get("implication_loop", 0),
                    "coping": state.get("coping_loop", 0),
                    "normative": state.get("normative_loop", 0),
                },
                "elapsed_seconds": round(elapsed, 2),
                "input_tokens": input_tok,
                "output_tokens": output_tok,
            }

        except Exception as e:
            elapsed = time.time() - tweet_start
            print(f"  Error ({elapsed:.1f}s): {e}")
            result = {
                "index": comment["index"],
                "tweet_id": comment["tweet_id"],
                "comment": comment["text"],
                "gold_label": comment["gold_label"],
                "error": str(e),
                "elapsed_seconds": round(elapsed, 2),
            }

        results.append(result)

        # Save progressively
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    batch_elapsed = time.time() - batch_start
    print(f"\nBatch complete. Results saved to: {OUTPUT_FILE}")
    print(f"  Processed this run: {len(remaining)} tweets in {batch_elapsed:.1f}s")
    print(f"  Total in file: {len(results)} tweets")
    print(f"  Tokens this run: {total_input_tokens} in / {total_output_tokens} out")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    asyncio.run(run_batch())