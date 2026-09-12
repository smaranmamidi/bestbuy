#!/usr/bin/env python3
"""Deterministic validator for output.csv. Exits non-zero on failure."""
import csv
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from finance import Data, parse_messages_for_user, build_user_forecast, simulate, parse_date, expand_option

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output.csv"

STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}


def main():
    data = Data(str(REPO / "dataset"))
    reqs = list(csv.DictReader(open(REPO / "dataset" / "requests.csv")))
    outs = list(csv.DictReader(open(OUT)))
    assert len(outs) == 250, f"expected 250 rows, got {len(outs)}"
    assert [r["request_id"] for r in outs] == [r["request_id"] for r in reqs], "order must match requests.csv"
    cols = ["request_id", "amount_safe_to_pay", "affordability_status", "recommended_payment_method",
            "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed", "decision_explanation"]
    with open(OUT) as f:
        header = f.readline().strip().split(",")
    assert header == cols, f"bad columns {header}"
    errors = []
    for req, o in zip(reqs, outs):
        rid = req["request_id"]
        try:
            requested = Decimal(req["requested_amount"].replace(",", ""))
            safe = Decimal(o["amount_safe_to_pay"].replace(",", ""))
            assert Decimal("0") - Decimal("0.01") <= safe <= requested + Decimal("0.01"), f"safe bounds {safe} vs {requested}"
            assert o["affordability_status"] in STATUSES, "bad status"
            assert o["recommended_payment_method"] in METHODS, "bad method"
            # payment plan parse
            plan = []
            if o["payment_plan"] != "none":
                for part in o["payment_plan"].split("|"):
                    d, a = part.split(":")
                    plan.append((date.fromisoformat(d), Decimal(a)))
                assert plan == sorted(plan), "plan not chronological"
            # earliest
            earliest = None
            if o["spending_changes_needed"] not in ("none", ""):
                parts = o["spending_changes_needed"].split("|")
                assert len(parts) <= 3, "max 3 changes"
                seen = {}
                for p in parts:
                    assert p.startswith("stop:") or p.startswith("reduce_to:"), f"bad change {p}"
            if o["earliest_date_for_full_payment"].strip():
                earliest = date.fromisoformat(o["earliest_date_for_full_payment"].strip())
            # status/method consistency
            if o["affordability_status"] == "affordable_now":
                assert o["recommended_payment_method"] == "full_payment", "now->full"
                assert earliest == date.fromisoformat(req["request_date"]), "now earliest=request_date"
            if o["recommended_payment_method"] == "partial_payment":
                assert o["affordability_status"] == "affordable_with_plan"
                assert req["allows_partial_payment"].lower() == "true"
                assert len(plan) == 2, "partial exactly 2"
                assert abs((plan[0][1] + plan[1][1]) - requested) < Decimal("0.02"), "partial sum"
                assert abs(plan[0][1] - safe) < Decimal("0.02"), "partial first=safe"
            if o["recommended_payment_method"] == "installments":
                # must match a supplied option exactly
                opts = data.options_by_request.get(rid, [])
                matched = False
                for opt in opts:
                    if opt["payment_method"] != "installments":
                        continue
                    exp = expand_option(opt)
                    if len(exp) == len(plan) and all(e[0] == p[0] and abs(e[1] - p[1]) < Decimal("0.02") for e, p in zip(exp, plan)):
                        matched = True
                assert matched, f"installment must match option for {rid}"
            if o["recommended_payment_method"] == "not_recommended":
                assert o["payment_plan"] == "none" and o["affordability_status"] == "not_affordable"
            # min balance check with engine forecast
            profile = data.profiles[req["user_id"]]
            home = profile["home_currency"]
            bal = Decimal(profile["current_available_balance"].replace(",", ""))
            minimum = Decimal(profile["minimum_balance_to_keep"].replace(",", ""))
            facts = parse_messages_for_user(data.messages_by_user.get(req["user_id"], []), rid)
            forecast, _ = build_user_forecast(data, req["user_id"], date.fromisoformat(req["request_date"]), home, facts)
            # parse spending (rebuild variable-category monthly credits like the engine)
            stop_ids, red = set(), {}
            if o["spending_changes_needed"] not in ("none", ""):
                for p in o["spending_changes_needed"].split("|"):
                    if p.startswith("stop:"):
                        stop_ids.add(p[5:])
                    else:
                        _, eid, amt = p.split(":")
                        red[eid] = Decimal(amt)
            var_per_day = Decimal("0")
            if stop_ids or red:
                fixed_cats = forecast.get("fixed_cats", set())
                for eid in stop_ids:
                    ev = data.events_by_id.get(eid)
                    if ev is not None and ev["category"] not in fixed_cats:
                        sd = parse_date(ev["settlement_date"])
                        amt = data.event_amount_home(ev, home, sd)
                        if amt is not None:
                            var_per_day += amt / Decimal("30")
                for eid, new_amt in red.items():
                    ev = data.events_by_id.get(eid)
                    if ev is not None and ev["category"] not in fixed_cats:
                        sd = parse_date(ev["settlement_date"])
                        amt = data.event_amount_home(ev, home, sd)
                        if amt is not None:
                            var_per_day += (amt - new_amt) / Decimal("30")
            ok, _ = simulate(date.fromisoformat(req["request_date"]), bal, minimum, forecast, plan, stop_ids, red, var_per_day)
            if not ok:
                # if baseline with no payments also fails, not_affordable is acceptable
                ok_base, _ = simulate(date.fromisoformat(req["request_date"]), bal, minimum, forecast, [], set(), {})
                if not (o["recommended_payment_method"] == "not_recommended" and not ok_base):
                    assert ok, f"min balance violated for {rid}"
        except AssertionError as e:
            errors.append(f"{rid}: {e}")
    if errors:
        print("VALIDATION FAILED")
        for e in errors[:30]:
            print(" ", e)
        sys.exit(1)
    print(f"Validation passed for {len(outs)} rows.")


if __name__ == "__main__":
    main()
