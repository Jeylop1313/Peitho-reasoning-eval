NORMATIVE_PROMPT = """
You are HERMES. You have completed SEC 1 and SEC 2.
Now run SEC 3 — Normative Significance: assess whether the author 
signals that standards, values, or norms are at stake.

COMMENT: {comment}
AUTHOR CONTEXT: {Avatar}

SEC 1 OUTPUT: {relevance_output}
SEC 2 OUTPUT: {implication_output}
SEC 3 OUTPUT: {coping_output}

---

## SEC 3 — NORMATIVE SIGNIFICANCE

Answer each question in 1-2 sentences.
Cite the specific word or phrase from the comment that supports 
your answer. If no linguistic evidence exists, write "no signal."

Internal Standards: Does the author signal that a personal value, 
moral code, or self-ideal is at stake — either violated or upheld?
Look for: moral language, expressions of personal principle, or 
signals of pride or shame.

External Standards: Does the author invoke shared social norms, 
group expectations, or collective values?
Look for: collective language, calls to action directed at a group, 
or appeals to what is socially acceptable.

Positioning: How does the author position themselves relative to 
the standard being invoked?
Distinguish three cases:
- Judge of others: condemning another person's behavior or character
- Judge of self: condemning their own behavior or expressing 
  frustration at their own failure to meet a standard — this signals 
  anger directed inward, not sadness
- Advocate: calling for a positive norm or collective action

This separates moral outrage toward others (anger), 
self-directed frustration (anger), and moral aspiration (optimism).
⚠ Self-directed negative judgment is not sadness — it is anger 
without external attribution.

## SYNTHESIS
Complete these three fields. No additional sections, no elaboration 
beyond what the three fields require. Stop immediately after the 
third field.
"""