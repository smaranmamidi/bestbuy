#!/usr/bin/env python3
"""Entry point: python3 code/main.py — reads dataset/, writes output.csv at repo root."""
import csv
import sys
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from finance import (Data, parse_messages_for_user, build_user_forecast, simulate,
                     max_safe_today, earliest_full_date, expand_option, fmt_amount,
                     parse_date, dec)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET = REPO_ROOT / "dataset"
OUT_PATH = REPO_ROOT / "output.csv"


def norm_methods(s):
    return set(x.strip() for x in (s or "").split("|") if x.strip())


def decide_one(data, req):
    user_id = req["user_id"]
    request_id = req["request_id"]
    request_date = parse_date(req["request_date"])
    desired = parse_date(req["desired_completion_date"])
    requested = Decimal(req["requested_amount"].replace(",", ""))
    allows_partial = (req.get("allows_partial_payment") or "").strip().lower() == "true"
    profile = data.profiles[user_id]
    home = profile["home_currency"]
    start_balance = Decimal(profile["current_available_balance"].replace(",", ""))
    minimum = Decimal(profile["minimum_balance_to_keep"].replace(",", ""))
    consider = norm_methods(profile.get("payment_methods_user_will_consider", ""))
    try:
        max_inst = int(profile.get("max_installment_months") or "0") if (profile.get("max_installment_months") or "").strip() else None
    except Exception:
        max_inst = None

    msgs = data.messages_by_user.get(user_id, [])
    facts = parse_messages_for_user(msgs, request_id)
    forecast, flex_templates = build_user_forecast(data, user_id, request_date, home, facts)

    # Adjust start balance for pending debits settling on/before request_date? None in data typically.
    # Reserve pending debits with settlement in future via explicit flows (handled in simulate).

    safe = max_safe_today(request_date, start_balance, minimum, forecast, requested)
    if safe < 0:
        safe = Decimal("0")
    if safe > requested:
        safe = requested
    earliest = earliest_full_date(request_date, start_balance, minimum, forecast, requested)

    options = data.options_by_request.get(request_id, [])
    # candidate evaluation
    candidates = []  # (rank_tuple, method, plan, spending, status)

    def spending_str(stop_ids, reduce_map):
        if not stop_ids and not reduce_map:
            return "none"
        parts = []
        for eid in sorted(stop_ids):
            parts.append(f"stop:{eid}")
        for eid in sorted(reduce_map):
            parts.append(f"reduce_to:{eid}:{fmt_amount(reduce_map[eid])}")
        return "|".join(parts)

    # 1. full payment
    if "full_payment" in consider:
        ok, _ = simulate(request_date, start_balance, minimum, forecast, [(request_date, requested)])
        if ok:
            candidates.append(("full", Decimal(requested), request_date, 1, "payment_option_full",
                               "full_payment", [(request_date, requested)], set(), {}, "affordable_now", []))

    # 2. partial payment
    if allows_partial and "partial_payment" in consider and Decimal("0") < safe < requested and earliest is not None and earliest <= desired:
        rem = requested - safe
        ok, _ = simulate(request_date, start_balance, minimum, forecast,
                         [(request_date, safe), (earliest, rem)])
        if ok:
            candidates.append(("partial", requested, request_date, 2, "partial",
                               "partial_payment", [(request_date, safe), (earliest, rem)], set(), {}, "affordable_with_plan", []))

    # 3. installments
    for opt in options:
        if opt["payment_method"] != "installments":
            continue
        if "installments" not in consider:
            continue
        n = int(opt["number_of_payments"] or "1")
        freq = (opt.get("payment_frequency_days") or "").strip()
        freq_days = int(freq) if freq else 0
        if max_inst is not None:
            # duration in months, not raw payment count
            if freq_days:
                months = (n - 1) * freq_days / 30.0 + 1
            else:
                months = 1 if n == 1 else n
            if months > max_inst + 1e-9:
                continue
        pays = expand_option(opt)
        # must complete by desired
        if pays[-1][0] > desired:
            continue
        # all payments within forecast? first must be >= request_date
        if pays[0][0] < request_date:
            continue
        ok, _ = simulate(request_date, start_balance, minimum, forecast, pays)
        if ok:
            total = Decimal(opt["total_payable_amount"].replace(",", ""))
            candidates.append(("inst", total, pays[0][0], len(pays), opt["payment_option_id"],
                               "installments", pays, set(), {}, "affordable_with_plan", []))

    # 4. wait (must complete by desired_completion_date per spec)
    wait_plan = None
    if earliest is not None and "full_payment" in consider and earliest <= desired:
        ok, _ = simulate(request_date, start_balance, minimum, forecast, [(earliest, requested)])
        if ok and earliest > request_date:
            wait_plan = [(earliest, requested)]
            candidates.append(("wait", requested, earliest, 1, "wait",
                               "wait", wait_plan, set(), {}, "affordable_later", []))

    # 5. flexible spending plans (try to make full safe by desired)
    flex_cands = []
    if flex_templates:
        # filter by user willingness
        prot = set(x.strip() for x in (profile.get("expense_categories_to_protect") or "").split("|") if x.strip())
        can_reduce = set(x.strip() for x in (profile.get("expense_categories_user_is_willing_to_reduce") or "").split("|") if x.strip())
        can_stop = set(x.strip() for x in (profile.get("expense_categories_user_is_willing_to_stop") or "").split("|") if x.strip())
        usables = []
        for e in flex_templates:
            cat = e["category"]
            if cat in prot:
                continue
            flex = e["flexibility"]
            sd = parse_date(e["settlement_date"])
            amt_home = data.event_amount_home(e, home, sd)
            if amt_home is None:
                continue
            min_allowed = None
            if (e.get("minimum_allowed_amount") or "").strip():
                try:
                    min_allowed = Decimal(e["minimum_allowed_amount"].replace(",", ""))
                    # min_allowed is in event currency; convert
                    min_allowed = data.fx.convert(min_allowed, e["currency"], home, sd)
                except Exception:
                    min_allowed = None
            actions = []
            if flex in ("stoppable", "reducible_or_stoppable") and cat in can_stop:
                actions.append(("stop", None))
            if flex in ("reducible", "reducible_or_stoppable") and cat in can_reduce and min_allowed is not None and min_allowed < amt_home:
                actions.append(("reduce", min_allowed))
            if actions:
                usables.append((e, amt_home, actions))
        # sort by monthly savings desc, limit to top 6 to bound combos
        usables.sort(key=lambda x: -x[1])
        usables = usables[:6]
        # try combos: singles, pairs, triples; prefer smallest intervention
        import itertools
        tried = []
        # build action options per template
        for r in (1, 2, 3):
            for combo in itertools.combinations(usables, r):
                # for reducible_or_stoppable with both options, try both variants (limit)
                variants = [[]]
                for (e, amt, acts) in combo:
                    nv = []
                    for v in variants:
                        for kind, val in acts[:2]:
                            nv.append(v + [(e, kind, val)])
                    variants = nv[:4]
                for var in variants:
                    stop_ids = set(e["event_id"] for e, k, v in var if k == "stop")
                    red = {e["event_id"]: v for e, k, v in var if k == "reduce"}
                    # distinct events check
                    if len(stop_ids) + len(red) != len(var):
                        continue
                    tried.append((stop_ids, red))
            # evaluate level by level; stop early if found at this level
            found_this_level = []
            for stop_ids, red in tried:
                # full payment on request_date with changes?
                # Variable-category templates save every day via a lower burn
                # rate (monthly savings spread over 30 days).
                fixed_cats = forecast.get("fixed_cats", set())
                per_day = Decimal("0")
                tmpl_by_id = {e["event_id"]: (e, amt) for e, amt, _ in usables}
                for eid in stop_ids:
                    e, amt = tmpl_by_id[eid]
                    if e["category"] not in fixed_cats:
                        per_day += amt / Decimal("30")
                for eid, new_amt in red.items():
                    e, amt = tmpl_by_id[eid]
                    if e["category"] not in fixed_cats:
                        per_day += (amt - new_amt) / Decimal("30")
                ok, _ = simulate(request_date, start_balance, minimum, forecast,
                                 [(request_date, requested)], stop_ids, red, per_day)
                if ok:
                    # must use full_payment method eligible
                    if "full_payment" in consider:
                        total = requested
                        found_this_level.append((stop_ids, red))
            if found_this_level:
                # pick smallest savings (prefer reduce over stop, smaller count already minimal)
                def savings(sr):
                    sids, rd = sr
                    tot = Decimal("0")
                    for (e, amt, acts) in combo if False else []:
                        pass
                    return (len(sids) + len(rd),)
                # choose first (usables sorted); take minimal total reduced amount change?
                best = sorted(found_this_level, key=lambda x: (len(x[0]) + len(x[1])))[0]
                sids, rd = best
                # rebuild per-day savings for the winner
                fixed_cats = forecast.get("fixed_cats", set())
                tmpl_by_id = {e["event_id"]: (e, amt) for e, amt, _ in usables}
                best_per_day = Decimal("0")
                for eid in sids:
                    e, amt = tmpl_by_id[eid]
                    if e["category"] not in fixed_cats:
                        best_per_day += amt / Decimal("30")
                for eid, new_amt in rd.items():
                    e, amt = tmpl_by_id[eid]
                    if e["category"] not in fixed_cats:
                        best_per_day += (amt - new_amt) / Decimal("30")
                flex_cands.append(("flex", requested, request_date, 1, "flex",
                                   "full_payment", [(request_date, requested)], sids, rd, "affordable_with_plan", best_per_day))
                break
            tried = []
        # end flex

    candidates.extend(flex_cands)

    # ranking among eligible safe that complete by desired (except wait which completes on earliest<=? wait completes on earliest which may be <= desired? sample wait plans complete on earliest which equals desired? Actually wait earliest can equal desired.)
    # Filter to those completing by desired, except flex already by request_date
    def completes_by_desired(c):
        plan = c[6]
        return plan[-1][0] <= desired

    # rank key: (completes, no_spending, total, start, fewer, option_id)
    def rank_key(c):
        _tag, total, start, npay, optid, method, plan, sids, rd, status, _credits = c
        completes = 0 if plan[-1][0] <= desired else 1
        nospend = 0 if (not sids and not rd) else 1
        return (completes, nospend, total, start, npay, optid)

    winner = None
    if candidates:
        # only consider those completing by desired first; if none, fallback
        completing = [c for c in candidates if c[6][-1][0] <= desired]
        pool = completing if completing else candidates
        # Affordable_now only if full safe today and method full; else with_plan/later
        pool_sorted = sorted(pool, key=rank_key)
        winner = pool_sorted[0]

    if winner is None:
        # No safe eligible plan completing by desired. Wait only if earliest completes by desired.
        if earliest is not None and "full_payment" in consider and earliest <= desired:
            if earliest <= request_date + timedelta(days=FORECAST_DAYS):
                # check if wait plan safe (already verified in earliest calc)
                status, method = "affordable_later", "wait"
                plan = [(earliest, requested)]
                sp = "none"
                expl = make_explanation(home, start_balance, minimum, safe, requested, method, plan, earliest, sp, forecast)
                return out_row(request_id, safe, requested, status, method, plan, earliest, sp, expl)
        status, method, plan, earliest_out, sp = "not_affordable", "not_recommended", [], None, "none"
        expl = make_explanation(home, start_balance, minimum, safe, requested, method, plan, earliest, sp, forecast)
        return out_row(request_id, safe, requested, status, method, plan, earliest_out, sp, expl)

    _tag, _total, _start, _npay, _optid, method, plan, sids, rd, status, _credits = winner
    sp = spending_str(sids, rd) if (sids or rd) else "none"
    # earliest output is independent capacity measure
    earliest_out = earliest
    expl = make_explanation(home, start_balance, minimum, safe, requested, method, plan, earliest_out, sp, forecast)
    return out_row(request_id, safe, requested, status, method, plan, earliest_out, sp, expl)


def out_row(request_id, safe, requested, status, method, plan, earliest, spending, expl):
    if safe < 0:
        safe = Decimal("0")
    if safe > requested:
        safe = requested
    if plan:
        plan_s = "|".join(f"{d.isoformat()}:{fmt_amount(a)}" for d, a in sorted(plan))
    else:
        plan_s = "none"
    e_s = earliest.isoformat() if earliest is not None else ""
    return {
        "request_id": request_id,
        "amount_safe_to_pay": fmt_amount(safe),
        "affordability_status": status,
        "recommended_payment_method": method,
        "payment_plan": plan_s,
        "earliest_date_for_full_payment": e_s,
        "spending_changes_needed": spending if spending else "none",
        "decision_explanation": expl,
    }


def make_explanation(home, bal, minimum, safe, requested, method, plan, earliest, spending, forecast):
    if method == "full_payment" and not spending or spending == "none":
        if plan and plan[0][0] is not None:
            return f"Pay {home} {fmt_amount(requested)} today. This keeps the {home} {fmt_amount(minimum)} minimum protected over the next 90 days."
    if method == "installments" and plan:
        per = fmt_amount(plan[0][1])
        return f"Use {len(plan)} installments of {home} {per}, starting {plan[0][0].isoformat()}. This keeps the {home} {fmt_amount(minimum)} minimum protected."
    if method == "partial_payment" and plan and len(plan) == 2:
        return f"Pay {home} {fmt_amount(plan[0][1])} today and the remaining {home} {fmt_amount(plan[1][1])} on {plan[1][0].isoformat()}. This completes the full request and keeps the {home} {fmt_amount(minimum)} minimum protected."
    if method == "wait" and plan:
        return f"Pay {home} {fmt_amount(requested)} in full on {plan[0][0].isoformat()}. Paying earlier would take the balance below the {home} {fmt_amount(minimum)} minimum."
    if method == "full_payment" and spending != "none":
        return f"Pay {home} {fmt_amount(requested)} today with spending change {spending}. This keeps the {home} {fmt_amount(minimum)} minimum protected."
    if method == "not_recommended":
        return f"Do not make this payment. None of the available options keeps the {home} {fmt_amount(minimum)} minimum protected."
    return f"Decision {method} for {home} {fmt_amount(requested)} with {home} {fmt_amount(safe)} safe today."


def main():
    data = Data(str(DATASET))
    rows = []
    for req in data.requests:
        try:
            rows.append(decide_one(data, req))
        except Exception as ex:
            requested = Decimal(req["requested_amount"].replace(",", ""))
            rows.append({
                "request_id": req["request_id"],
                "amount_safe_to_pay": "0",
                "affordability_status": "not_affordable",
                "recommended_payment_method": "not_recommended",
                "payment_plan": "none",
                "earliest_date_for_full_payment": "",
                "spending_changes_needed": "none",
                "decision_explanation": f"Could not complete evaluation, so no payment is recommended. Error noted during processing.",
            })
            print(f"WARN {req['request_id']}: {ex}", file=sys.stderr)
    cols = ["request_id", "amount_safe_to_pay", "affordability_status", "recommended_payment_method",
            "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed", "decision_explanation"]
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
