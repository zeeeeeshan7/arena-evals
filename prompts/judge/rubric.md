# Grading rubric (v1)

Grade the assistant answer on two independent criteria.

**correct**: true if the answer states the same facts as the reference answer. Wording, order and extra harmless detail do not matter. Numbers must match the reference (units may be written differently). A missing required fact, a wrong value, or a contradiction makes it false. Hedging between two values is false.

**faithful**: true if every factual claim in the answer is supported by the text of the cited documents shown below. A claim that is true but not supported by any cited document makes it false. If no valid documents were cited, faithful is false.

Length is not a criterion. Do not reward longer answers or penalise short ones.

Reply with exactly this JSON object and nothing else:

{"correct": true or false, "faithful": true or false, "reason": "one or two sentences"}
