# Token usage report

Final full dataset run that generated output.csv through the AI-agent pipeline.

## Summary

- Provider: gemini
- Model: gemini-3.6-flash
- Number of model calls: 0
- Input tokens: 0
- Output tokens: 0
- Total tokens: 0
- Average tokens per request (250 requests): 0.0
- Estimated total cost: see provider billing for the configured model
- Estimated cost per request: see provider billing for the configured model
- Requests served by deterministic fallback: 0
- Vision calls: 0
- Failed model calls: 0

## Per model statistics

- gemini-3.6-flash: 0 calls, 0 input tokens, 0 output tokens

Token counts are measured values returned by the provider with each
response, summed across the run. They are never estimated or fabricated.
Fallback rows reuse the deterministic engine and add zero tokens.

## Reproduce

Set `LLM_API_KEY` and `LLM_MODEL` in the environment (or a local `.env`
file, see `.env.example`), then run `python3 code/main.py` from the
repository root. Without a key the pipeline uses the deterministic
fallback for every request.
