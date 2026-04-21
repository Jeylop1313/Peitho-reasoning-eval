IMPLICATIONS_PROMPT = """
You are HERMES. You have completed SEC 1 — Relevance.
Now run SEC 2 — Implication: assess what the author believes 
this event means for them.

COMMENT: {comment}
AUTHOR CONTEXT: {Avatar}

SEC 1 OUTPUT:
{relevance_output}

---

## SEC 2 — IMPLICATION

Answer each question in 1-2 sentences.
Cite the specific word or phrase from the comment that supports 
your answer. If no linguistic evidence exists, write "no signal."

Causal Attribution: Who or what does the author hold responsible 
for this situation — themselves, another person, an institution, 
chance, or no one? Look for: direct blame markers ("you", "they", 
"he/she"), accusatory framing, named agents, or the absence of 
any external target.

Goal Conduciveness: Does the author signal that this event helps 
or blocks something they care about — or that it opens or closes 
a possibility? Look for: achievement markers ("finally", "managed", 
"got"), obstruction markers ("can't", "won't", "blocked", "lost"), 
or forward-looking possibility markers ("maybe", "could", "going to").

Discrepancy from Expectation: Does the author signal that this 
outcome is different from what they expected — better, worse, 
or simply surprising? Look for: contrast markers ("but", "yet", 
"still", "even though", "instead"), explicit surprise, or 
expressions of ironic resignation.

Urgency: Does the author signal that this requires immediate 
attention or action — or that time pressure is a factor? 
If no signal, write "no signal" and move on.

## SYNTHESIS
In 2-3 sentences: what does the implication assessment add to 
what SEC 1 already found? Specifically state:
- CAUSAL_AGENT: [self / other / institution / chance / absent]
- CONDUCIVENESS: [facilitative / obstructive / neutral]
- IMPLICATION_SIGNAL: [the single strongest marker found]

Stop after SYNTHESIS. Do not add sections or commentary beyond 
what is requested above.
"""