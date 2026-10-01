## Output contract (required)

End your final message with exactly one line in this form, with nothing after it:

FINAL: {"answer": "<answer as a string>", "citations": ["<doc id>"], "abstain": <true or false>}

- `answer` is always a JSON string, including numbers (write "4475 USD", not 4475).
- `citations` lists the IDs of the documents that support the answer. Use [] when abstaining.
- `abstain` is true only when the documents do not contain the answer. Then `answer` is "".

Example:
FINAL: {"answer": "16 weeks", "citations": ["HR-009"], "abstain": false}
