# Token usage report

Final full dataset run that generated output.csv. The run used no external model calls. All financial arithmetic, message parsing, image amount extraction, forecasting, ranking, and explanations are deterministic and local.

## Summary

- Provider: none (local deterministic engine)
- Models used: none
- Number of model calls: 0
- Input tokens: 0
- Output tokens: 0
- Total tokens: 0
- Average tokens per request (250 requests): 0
- Estimated total cost: USD 0.00
- Estimated cost per request: USD 0.00

## Per model statistics

No models were called, so there are no per model rows. Image amounts came from local OCR with tesseract plus a manual cache of 16 values stored in code/finance.py. Message facts came from local regular expressions, including Indonesian payroll patterns and single-cycle temporary pay scoping. Explanations are template based and grounded in computed balances.

## Reproduce

Run `python3 code/main.py` from the repository root. It reads dataset/ and writes output.csv with 250 rows. Token counts remain zero on every rerun because the code path makes no network calls.
