# Buy or Wait solution

Deterministic financial engine with local message and image handling. No network calls.

## Run

From the repository root:

python3 code/main.py

This reads dataset/ and writes output.csv with 250 rows.

## Validate

python3 code/validate.py

## Tests

python3 -m unittest discover -s code/tests -v

## Sample check

python3 code/evaluation/evaluate.py

## Layout

- code/main.py: entry point and plan ranking
- code/finance.py: loader, exchange conversion, message rules, image cache, forecast, simulation
- code/validate.py: deterministic output checks
- code/tests/test_engine.py: 20 unit tests
- code/evaluation/evaluate.py: sample scoring
- code/evaluation/usage_report.md: token report for the final run
- evaluation/usage_report.md: same report at repo level
- prompts/prompts.md: templates and parsing notes
- output.csv: predictions for dataset/requests.csv
