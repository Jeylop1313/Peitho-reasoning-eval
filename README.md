# PEITHO: A Cognitive Agent with LLMs for Sentiment Classification and Interpretability in Social Media Sarcasm

Peitho (formerly HERMES) is a cognitive appraisal agent for sentiment classification grounded in Scherer's Component Process Model (CPM). It implements Theory-Driven Chain-of-Thought (TD-CoT) reasoning via a multi-stage pipeline built with LangGraph and Llama 4 Scout (via Groq).

## Architecture

This repository contains the **TASS 2019 PE (Spanish)** version of Peitho. The SemEval-2017 (English) version lives in [Peitho](https://github.com/Jeylop1313/Peitho).

Peitho processes each tweet through 4 sequential Stimulus Evaluation Checks (SECs), followed by a convergence node that produces the final sentiment label:

```
Relevance ⇄ Tools → Implication → Coping → Normative → Convergence → END
```

Each SEC analyzes a different dimension of cognitive appraisal as defined by CPM:

- **SEC 1 — Relevance**: Novelty, familiarity, intrinsic pleasantness, goal relevance. Has access to web search tools for context verification.
- **SEC 2 — Implication**: Causal attribution, goal conduciveness, discrepancy from expectation, urgency.
- **SEC 3 — Coping**: Control, power, adjustment potential.
- **SEC 4 — Normative**: Internal and external standards, self-positioning.
- **Convergence**: Integrates the 4 SEC outputs into a final sentiment label (P / N / NEU / NONE).

## Benchmark

- **Dataset**: TASS 2019, Peru subset (train partition; labels P, N, NEU, NONE)
- **Primary metrics**: AvgRec, Macro-F1
- **Secondary metrics**: F1^PN, Accuracy, Micro-F1
- **Model**: Llama 4 Scout via Groq API

## Project Structure

```
src/enrichment_agent/
├── Graph.py              # LangGraph pipeline (4 SECs + Convergence)
├── State.py              # Agent state with token accumulators
├── configuration.py      # Runtime configuration
├── converge_tass2019PE.py # Convergence prompt (TASS 2019 PE, used here)
├── relevance.py          # SEC 1 prompt
├── implications.py       # SEC 2 prompt
├── coping.py             # SEC 3 prompt
├── normative.py          # SEC 4 prompt
├── tools.py              # Tavily search tools (SEC 1 only)
├── utils.py              # Model initialization
├── batch_runner.py       # Batch evaluation with checkpointing and retry
├── ablation.py           # Zero-shot and CoT baselines (no CPM pipeline)
├── benchmarks.py         # Metrics computation and error diagnostics
└── Datasets/             # Legacy SemEval files (not used by this version)
```

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/Jeylop1313/Peitho-reasoning-eval.git
cd Peitho-reasoning-eval
pip install -e .
```

### 2. Set up API keys

```bash
cp .env.example .env
```

Edit `.env` and add:

```
GROQ_API_KEY=your-groq-api-key
TAVILY_API_KEY=your-tavily-api-key
```

### 3. Run interactively (LangGraph Studio)

```bash
langgraph dev
```

### 4. Run batch evaluation

```bash
python -m enrichment_agent.batch_runner
```

Supports checkpointing — if interrupted, re-running the same command resumes from where it left off.

### 5. Generate diagnostics

```bash
python -m enrichment_agent.benchmarks
```

Produces a `diagnostics_output/` folder with metrics summary, confusion matrix, error traces, and per-pattern error files.

## Reproducibility

Settings used to produce the results reported in the paper.

**Model**
- `meta-llama/llama-4-scout-17b-16e-instruct`, served through the Groq API.
- Temperature 0 in every call.
- Maximum output tokens: 2048 per call in Peitho (SEC nodes and convergence); 1024 in the baselines.

**Peitho pipeline**
- Each SEC node runs a two-phase loop: phase 0 is exploratory reasoning; phase 1 forces the node to commit to a text output without tools.
- Only SEC 1 (Relevance) has tools: Tavily web search, `search_depth="basic"`, at most 3 results per query (`max_search_results` in `configuration.py`).
- The convergence node returns the label through a structured tool call (`ConvergenceInfo`). The prompt and label set are selected with `convergence_variant`; `batch_runner.py` sets it to `tass2019PE`.
- Output: `results_tass2019PE.json`, one record per tweet with the four SEC traces, the convergence reasoning and label, loop counts, time and token usage.

**Baselines (`ablation.py`)**
- One model call per tweet, no tools, no CPM pipeline.
- Zero-shot: task description and output format only.
- CoT: the zero-shot prompt plus "Pensemos paso a paso." No reasoning steps are prescribed.
- Select the condition and dataset with the `TECHNIQUE` and `DATASET` constants at the top of the file.

**Failures and retries**
- API calls are retried up to 3 times with exponential backoff on rate limits (429) and server errors (5xx).
- A tweet that still fails is stored without a label (`convergence` is empty) and is reported as skipped by `benchmarks.py`.

**Run dates**
- TASS 2019 PE: Peitho, zero-shot and CoT, May 2026.

**Determinism**
- Temperature 0 does not guarantee identical outputs from a hosted model, and Tavily results depend on the date of the search. Re-running the pipeline can therefore produce small differences; the released prediction files are the reference results.

## Key Features

- **Theory-Driven CoT**: Each SEC implements a specific cognitive appraisal dimension from Scherer's CPM, constraining the LLM's reasoning toward theoretically meaningful dimensions.
- **Checkpointing**: Batch runner saves progress after each tweet. Resume by re-running the same command.
- **Retry with backoff**: Exponential backoff on rate limits (429) and server errors (500).
- **Token tracking**: Accumulated across all nodes and loops via LangGraph reducers.
- **Error diagnostics**: Per-pattern error files with full SEC traces for qualitative analysis.

## Author

Yeison López — Psychology undergraduate at Universidad Nacional de Colombia (UNAL). Thesis project on NLP-based sentiment classification using cognitive appraisal theory.
