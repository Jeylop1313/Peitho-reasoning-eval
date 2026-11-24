# Reasoning evaluation

Code for the reasoning evaluation of Peitho on the 40 TASS 2019 PE comments used in the human study: trace processing, similarity analysis and dissociation analysis. The classification code that produces the predictions and traces lives in `src/enrichment_agent/`; the data-collection instruments live in [Peitho-estudio](https://github.com/Jeylop1313/Peitho-estudio).

## Stages and scripts

Scripts are added stage by stage; entries marked *pending* are not in the repository yet.

| Stage | Script | Purpose | Status |
|---|---|---|---|
| 4. Trace processing | `extract_cot_traces.py` | Extract the CoT reasoning of the 40 comments from the TASS CoT predictions | pending |
| 4. Trace processing | `summarize_traces.py` | Summarize Peitho and CoT traces (1 to 2 sentences, at most 40 words) with the same base model at temperature 0 | pending |
| 5. Similarity | `metric_selection/bertscore_eval.py` | Compare similarity metrics and select the embedding model | pending |
| 5. Similarity | `peitho_tables.py` | Similarity and discrimination tables, paired tests and TOST, with the comment as the unit of analysis | pending |
| 6. Dissociation | `dissociation.py` | Combine classification and similarity results | pending |

## Data

`data/` holds only material that can be shared: machine traces without the tweet text, and comment-level aggregated scores. Individual human traces are not distributed, as required by the informed consent of the study.
