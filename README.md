# PEITHO: A Cognitive Agent with LLMs for Sentiment Classification and Interpretability in Social Media Sarcasm

Peitho (Heuristic Emotion Recognition through Modular Evaluative Synthesis) is a cognitive appraisal agent for sentiment classification grounded in Scherer's Component Process Model (CPM). It implements Theory-Driven Chain-of-Thought (TD-CoT) reasoning via a multi-stage pipeline built with LangGraph and Llama 4 Scout (via Groq).

## Architecture

Peitho processes each tweet through 4 sequential Stimulus Evaluation Checks (SECs), followed by a convergence node that produces the final sentiment label:

```
Relevance ⇄ Tools → Implication → Coping → Normative → Convergence → END
```

Each SEC analyzes a different dimension of cognitive appraisal as defined by CPM:

- **SEC 1 — Relevance**: Novelty, familiarity, intrinsic pleasantness, goal relevance. Has access to web search tools for context verification.
- **SEC 2 — Implication**: Causal attribution, goal conduciveness, discrepancy from expectation, urgency.
- **SEC 3 — Coping**: Control, power, adjustment potential.
- **SEC 4 — Normative**: Internal and external standards, self-positioning.
- **Convergence**: Integrates the 4 SEC outputs into a final sentiment label (positive / negative / neutral).

## Benchmark

- **Dataset**: SemEval-2017 Task 4A (11,906 tweets — positive, negative, neutral)
- **Primary metrics**: AvgRec, Macro-F1
- **Secondary metrics**: F1^PN, Accuracy, Micro-F1
- **Model**: Llama 4 Scout via Groq API

## Project Structure

```
src/enrichment_agent/
├── Graph.py              # LangGraph pipeline (4 SECs + Convergence)
├── State.py              # Agent state with token accumulators
├── configuration.py      # Runtime configuration
├── converge_prompt.py    # Convergence prompt template
├── relevance.py          # SEC 1 prompt
├── implications.py       # SEC 2 prompt
├── coping.py             # SEC 3 prompt
├── normative.py          # SEC 4 prompt
├── tools.py              # Tavily search tools (SEC 1 only)
├── utils.py              # Model initialization
├── batch_runner.py       # Batch evaluation with checkpointing and retry
├── ablation.py           # Zero-shot baseline (no CPM pipeline)
├── benchmarks.py         # Metrics computation and error diagnostics
└── Datasets/             # SemEval datasets
```

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/Jeylop13/Peitho.git
cd Peitho
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

## Key Features

- **Theory-Driven CoT**: Each SEC implements a specific cognitive appraisal dimension from Scherer's CPM, constraining the LLM's reasoning toward theoretically meaningful dimensions.
- **Checkpointing**: Batch runner saves progress after each tweet. Resume by re-running the same command.
- **Retry with backoff**: Exponential backoff on rate limits (429) and server errors (500).
- **Token tracking**: Accumulated across all nodes and loops via LangGraph reducers.
- **Error diagnostics**: Per-pattern error files with full SEC traces for qualitative analysis.

## Author

Yeison López — Psychology undergraduate at Universidad Nacional de Colombia (UNAL). Thesis project on NLP-based sentiment classification using cognitive appraisal theory.
