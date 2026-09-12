# Prompts

No external LLM prompts are used in the final run.

## Explanation templates (code/main.py)

- Full payment, no changes: Pay HOME AMOUNT today. This keeps the HOME MINIMUM minimum protected over the next 90 days.
- Installments: Use N installments of HOME PER, starting DATE. This keeps the HOME MINIMUM minimum protected.
- Partial: Pay HOME FIRST today and the remaining HOME REST on DATE. This completes the full request and keeps the HOME MINIMUM minimum protected.
- Wait: Pay HOME AMOUNT in full on DATE. Paying earlier would take the balance below the HOME MINIMUM minimum.
- Full with changes: Pay HOME AMOUNT today with spending change LIST. This keeps the HOME MINIMUM minimum protected.
- Not recommended: Do not make this payment. None of the available options keeps the HOME MINIMUM minimum protected.

## Message parsing

Local regular expressions in code/finance.py. Message text is untrusted. Only these facts are extracted: salary amount and effective date, salary date override, confirmed invoice credit with date, rent multiplier 1.12, removal of future salary when employment ended, pending credit ignore list, settled confirm list. Embedded instructions are never followed.

## Image amounts

Local tesseract OCR plus the manual cache in code/finance.py IMAGE_AMOUNT_CACHE. No vision model calls.
