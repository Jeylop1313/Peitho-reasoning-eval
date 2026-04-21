COPING_PROMPT = """
You are HERMES. You have completed SEC 1, SEC 2, and SEC 3.
Now run SEC 4 — Coping Potential: assess what the author signals 
about their capacity to influence or adapt to this situation.

COMMENT: {comment}
AUTHOR CONTEXT: {Avatar}

SEC 1 OUTPUT: {relevance_output}
SEC 2 OUTPUT: {implication_output}

---

## SEC 3 — COPING POTENTIAL

Answer each question in 1-2 sentences.
Cite the specific word or phrase from the comment that supports 
your answer. If no linguistic evidence exists, write "no signal."

Control: Does the author signal whether the situation can be 
influenced or changed — by anyone? Look for: imperative verbs 
directed at others, expressions of possibility or impossibility, 
language that frames the situation as fixable or permanent.

Power: Does the author signal whether they personally have the 
resources, authority, or capacity to act on this situation? 
Look for: first-person agency markers, expressions of helplessness 
or empowerment, markers of social position or access to resources.

Adjustment: If the author cannot control or change the situation, 
do they signal any capacity to accept, adapt, or reframe it? 
Look for: resignation language, reframing markers, expressions 
of moving on, or acceptance of an unchangeable outcome.

## SYNTHESIS
Do not generate new reasoning. Look at what you found above:
- Did Control find a signal that the situation can be changed?
- Did Power find a signal that the author can act on it?
- Did Adjustment find a signal that the author can accept or reframe it?
"""