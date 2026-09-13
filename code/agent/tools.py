"""Agent tools: thin, validated wrappers around the deterministic engine.

Every tool calls the existing implementation in finance.py / main.py.
No financial logic is duplicated here. Amounts cross tool boundaries as
strings; dates as YYYY-MM-DD. Tools never invent payment options,
spending changes, income, or expenses.
"""
from datetime import date
from decimal import Decimal

from finance import (parse_date, fmt_amount, simulate, max_safe_today,
                     earliest_full_date, expand_option, parse_messages_for_user,
                     ocr_image_amount, extract_amount_from_text,
                     IMAGE_AMOUNT_CACHE)


def _d(x):
    return fmt_amount(x)


def _iso(d):
    return d.isoformat() if d is not None else ""


def get_financial_profile(data, user_id):
    """Current balance, currency, minimum, preferences. Read-only."""
    p = data.profiles[user_id]
    return {"user_id": user_id,
            "home_currency": p["home_currency"],
            "current_available_balance": str(p["current_available_balance"]),
            "minimum_balance_to_keep": str(p["minimum_balance_to_keep"]),
            "financial_priorities": p.get("financial_priorities", ""),
            "expense_categories_to_protect": p.get("expense_categories_to_protect", ""),
            "expense_categories_user_is_willing_to_reduce": p.get(
                "expense_categories_user_is_willing_to_reduce", ""),
            "expense_categories_user_is_willing_to_stop": p.get(
                "expense_categories_user_is_willing_to_stop", ""),
            "payment_methods_user_will_consider": p.get(
                "payment_methods_user_will_consider", ""),
            "max_installment_months": p.get("max_installment_months", "")}


def get_financial_history(data, user_id, request_date, home):
    """Settled past income/expenses relevant to forecasting. Read-only."""
    import datetime as _dt
    from collections import Counter
    rd = parse_date(request_date) if isinstance(request_date, str) else request_date
    start = rd - _dt.timedelta(days=120)
    income, expense = [], []
    cats = Counter()
    for e in data.events_by_user.get(user_id, []):
        sd = parse_date(e["settlement_date"])
        if sd is None or sd > rd or sd < start or e["status"] != "settled":
            continue
        amt = data.event_amount_home(e, home, sd)
        if amt is None:
            continue
        rec = {"event_id": e["event_id"], "category": e["category"],
               "direction": e["direction"], "amount": _d(amt),
               "settlement_date": sd.isoformat()}
        if e["direction"] == "credit":
            income.append(rec)
        elif e["direction"] == "debit":
            expense.append(rec)
            cats[e["category"]] += 1
    return {"income_records": income[-12:], "income_count": len(income),
            "expense_records": expense[-12:], "expense_count": len(expense),
            "expense_categories": dict(cats)}


def get_pending_and_future_commitments(data, ctx):
    """Pending debits, scheduled/confirmed future flows in the 90-day window."""
    out = []
    for sd, signed, eid, cat in ctx["forecast"]["explicit"]:
        out.append({"settlement_date": sd.isoformat(),
                    "signed_amount": _d(signed),
                    "event_id": eid, "category": cat})
    return {"flows": out, "count": len(out)}


def search_messages(data, user_id, request_id):
    """Messages for this user/request plus deterministically validated facts."""
    msgs = [m for m in data.messages_by_user.get(user_id, [])
            if not m["request_id"] or m["request_id"] == request_id]
    facts = parse_messages_for_user(data.messages_by_user.get(user_id, []), request_id)
    clean = [{"message_id": m["message_id"], "sent_at": m["sent_at"],
              "source_type": m["source_type"],
              "related_event_id": m.get("related_event_id", ""),
              "message_text": m["message_text"]} for m in msgs]
    return {"messages": clean,
            "validated_facts": {
                "salary_override": [str(facts["salary_override"][0]),
                                    facts["salary_override"][1]] if facts["salary_override"] else None,
                "salary_date_override": _iso(facts["salary_date_override"]),
                "rent_multiplier": str(facts["rent_multiplier"]) if facts["rent_multiplier"] else None,
                "remove_future_salary": facts["remove_future_salary"],
                "one_time_credits": len(facts["one_time_credits"])}}


def analyze_image(data, image_row):
    """Deterministic image interpretation: cached amount plus OCR text excerpt.

    Never invents values. When the linked event already has an amount, that
    amount stands; the image only fills genuinely blank amounts.
    """
    image_id = image_row["image_id"]
    ev = data.events_by_id.get(image_row.get("related_event_id", ""))
    result = {"image_id": image_id,
              "related_event_id": image_row.get("related_event_id", ""),
              "event_has_amount": bool(ev and (ev.get("amount") or "").strip()),
              "cached_amount": IMAGE_AMOUNT_CACHE.get(image_id),
              "ocr_excerpt": ""}
    if ev is not None and not (ev.get("amount") or "").strip():
        path = data.base / "media" / "images" / f"{image_id}.png"
        if path.exists():
            text = ocr_image_amount(str(path)) or ""
            found = extract_amount_from_text(text)
            result["ocr_excerpt"] = text[:400].replace("\n", " ")
            result["ocr_amount"] = _d(found) if found is not None else None
    return result


def convert_currency(data, amount, from_currency, to_currency, settle_date):
    """Deterministic conversion using challenge-provided dated rates only."""
    sd = parse_date(settle_date) if isinstance(settle_date, str) else settle_date
    out = data.fx.convert(Decimal(str(amount)), from_currency, to_currency, sd)
    return {"converted_amount": _d(out), "from": from_currency, "to": to_currency,
            "rate_date_used": sd.isoformat()}


def forecast_balance(data, ctx, payments, stop_ids=None, reduce_map=None, per_day=None):
    """90-day deterministic simulation. Returns verdict plus minimum point."""
    from decimal import Decimal as _D
    pays = [(parse_date(d) if isinstance(d, str) else d, _D(str(a)))
            for d, a in payments]
    red = {k: _D(str(v)) for k, v in (reduce_map or {}).items()}
    ok, mb = simulate(ctx["request_date"], ctx["start_balance"], ctx["minimum"],
                      ctx["forecast"], pays, set(stop_ids or []), red,
                      per_day if per_day is None else _D(str(per_day)))
    return {"safe": ok, "minimum_projected_balance": _d(mb),
            "minimum_required": _d(ctx["minimum"])}


def calculate_safe_amount(data, ctx):
    """Maximum payable today before optional spending changes."""
    out = max_safe_today(ctx["request_date"], ctx["start_balance"], ctx["minimum"],
                         ctx["forecast"], ctx["requested"])
    return {"amount_safe_to_pay": _d(out)}


def find_earliest_affordable_date(data, ctx):
    """First date the full amount is safe as one payment, if any."""
    out = earliest_full_date(ctx["request_date"], ctx["start_balance"], ctx["minimum"],
                             ctx["forecast"], ctx["requested"])
    return {"earliest_date_for_full_payment": _iso(out)}


def describe_candidate(c):
    tag, total, start, npay, optid, method, plan, sids, rd, status, _cr = c
    return {"method": method, "status": status,
            "total_payable": _d(total), "start": start.isoformat(),
            "num_payments": npay, "payment_option_id": optid,
            "plan": [(d.isoformat(), _d(a)) for d, a in plan],
            "spending_stops": sorted(sids),
            "spending_reductions": {k: _d(v) for k, v in rd.items()}}


def describe_candidate(c):
    """JSON-safe view of one raw candidate tuple for traces and the model."""
    tag, total, start, npay, optid, method, plan, sids, rd, status, _cr = c
    return {"method": method, "status": status,
            "total_payable": _d(total), "start": start.isoformat(),
            "num_payments": npay, "payment_option_id": optid,
            "plan": [(d.isoformat(), _d(a)) for d, a in plan],
            "spending_stops": sorted(sids),
            "spending_reductions": {k: _d(v) for k, v in rd.items()}}


def evaluate_payment_options(data, ctx):
    """Standard candidates (full/partial/installments/wait), simulated.

    Returns raw engine tuples under 'raw' plus readable views. The raw
    tuples flow unchanged into rank_strategies.
    """
    import main as _main
    raw = _main.evaluate_standard_candidates(data, ctx)
    return {"raw": raw, "candidates": [describe_candidate(c) for c in raw]}


def evaluate_spending_changes(data, ctx):
    """Flexible-spending candidates, simulated. Flexible-only, max 3."""
    import main as _main
    raw = _main.search_flex_plans(data, ctx)
    return {"raw": raw, "candidates": [describe_candidate(c) for c in raw]}


def rank_strategies(raw_candidates, desired):
    """Deterministic ranking over raw engine candidates.

    The model may propose an order but may not override this result.
    """
    import main as _main
    d = parse_date(desired) if isinstance(desired, str) else desired
    winner = _main.rank_candidates(list(raw_candidates), d)
    if winner is None:
        return {"winner": None, "winner_raw": None}
    return {"winner": describe_candidate(winner), "winner_raw": winner}
