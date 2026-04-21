"""
HERMES — Batch Runner
========================
Loads tweets from SemEval_Dataset_Unido.xlsx (SemEval-2017 Task 4A format),
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

import openpyxl
from dotenv import load_dotenv
from enrichment_agent.Graph import graph

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_PATH = "enrichment_agent/Datasets/SemEval_Dataset_Unido.xlsx"
OUTPUT_FILE = "results_current.json"
AVATAR = "Anglophone Twitter user"
START_INDEX = 0
MAX_COMMENTS = 11906

VALID_LABELS = {"positive", "negative", "neutral"}

MAX_RETRIES = 3
BASE_DELAY = 2  # seconds, doubles each retry


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
            return await graph.ainvoke({
                "topic": topic,
                "perfil_avatar": avatar,
            })
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