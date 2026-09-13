# Buy or Wait? — An AI-powered personalized financial purchase decision agent

For every request in `dataset/requests.csv` the agent decides whether the
user should pay in full, pay partially, use installments, wait, or not
proceed. The decision is personalized: the same balance yields different
answers for different income patterns, commitments, preferences, and
minimum-balance constraints.

## How the agent works

```text
USER REQUEST
      |
AI FINANCIAL AGENT (code/agent/orchestrator.py)
      |-- interpret request (model, schema-validated)
      |-- get_financial_profile / history / commitments (tools)
      |-- search_messages / analyze_image, only when linked (tools)
      |-- forecast_balance / calculate_safe_amount / earliest date (tools)
      |-- evaluate_payment_options (tools)
      |-- evaluate_spending_changes, only when full is unsafe (tools)
      |-- rank_strategies — deterministic rules, model cannot override
      |-- explain with grounded numbers (model) or template fallback
      |-- deterministic validation, fallback to engine on any failure
      |
output.csv
```

The agent is the orchestration layer. It decides what information is
needed and which tools to call. The deterministic engine
(`code/finance.py`, candidate logic in `code/main.py`) is the
authoritative computation layer for balances, forecasts, safe amounts,
installments, spending changes, dates, ranking, and final validation.

## Why the LLM does not perform financial arithmetic

Language models approximate text; money requires exactness. Every number
in the final row comes from deterministic tools. Model outputs are JSON
schema-validated, cross-checked against request records, and every number
in an explanation must already exist in the tool facts. Any disagreement
resolves in favor of the deterministic engine, and any model failure
falls back to it.

## How personalization works

Personalization comes from tool results, never invention: available
balance versus safe amount, rent/bill/debt obligations, confirmed versus
unstable income, recurring versus one-time spending, accepted payment
methods, willingness to stop or reduce flexible expenses, minimum
balance, and desired completion date. These factors select among
strategies and shape the explanation.

## Messages and images

Message text is untrusted data. Deterministic rules extract salary,
date, invoice, and rent facts; the model may add context notes but never
instructions. Images resolve through OCR plus a validated amount cache;
with `VISION_ENABLED=true` and a key set, a vision-capable model may
read the receipt for the current request, and its amount is accepted
only when consistent with the deterministic extraction.

## Payment strategies

Full, partial (two payments, totals match), installments (exact supplied
option schedules), wait (earliest safe date within the deadline), or not
proceeding. Ranking prefers deadline completion, no spending changes,
lowest total, earlier start, fewer payments, lowest option id.

## Fallback

Timeouts, auth failures, malformed JSON, invented values, or missing
configuration each trigger the deterministic fallback for that request,
counted in the usage report. Model failure can never produce an invalid
financial row.

## Configure the model

1. Create a local `.env` file (see `.env.example`).
2. Set `LLM_PROVIDER=gemini` (or `openai`).
3. Set `LLM_API_KEY=<your key>`.
4. Set `LLM_MODEL=gemini-3.6-flash` (Gemini) or `gpt-4o-mini` (OpenAI).
5. Run the project.

Never commit `.env` and never paste keys into source code. Without a key
the pipeline runs the deterministic fallback for every request.

## Run the project

```bash
python3 code/main.py        # agent pipeline -> output.csv + usage report
python3 code/validate.py    # deterministic output checks
python3 -m unittest discover -s code/tests -v
python3 code/evaluation/evaluate.py   # 25 solved samples
```

## Evaluation

Sample results: affordability 20/25, method 21/25, plan 19/25, earliest
17/25, spending changes 22/25, safe amount 11/25. Remaining gaps are
small calibration differences in recurring-spend estimation, documented
in prior audit notes. Full output passes `validate.py` with 250 rows.
