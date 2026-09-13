"""AI financial decision agent: orchestration over deterministic tools.

Pipeline per request:
  interpret -> gather context -> messages/images (conditional) ->
  forecast/safe/earliest -> payment options -> spending changes (conditional)
  -> deterministic rank -> grounded explanation -> validate (fallback safe).

The deterministic engine is the financial authority. Any disagreement
resolves in its favor. Any model failure falls back to decide_one.
"""
import sys
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import main as _main
from finance import FORECAST_DAYS, fmt_amount, parse_date

from agent import config as _config_mod
from agent import tools
from agent.model_client import ModelError, chat_complete, image_data_url
from agent.prompts import (EXPLAIN_SYSTEM, EXPLAIN_USER, INTERPRET_SYSTEM,
                           INTERPRET_USER, MESSAGE_SYSTEM, MESSAGE_USER)
from agent.schemas import (explanation_is_grounded, numbers_in_text,
                           parse_json_object, validate_final_result,
                           validate_interpretation)
from agent.usage import UsageTracker


def _trace():
    return {"steps": [], "tools": [], "fallbacks": 0, "overrides": 0}


def _log(trace, step):
    trace["steps"].append(step)


def _call_model(cfg, tracker, system, user, vision_data_urls=None,
                max_tokens=None):
    """One measured model call. Raises ModelError on any problem."""
    if not cfg.is_configured():
        raise ModelError("model not configured")
    content_blocks = [{"type": "text", "text": user}]
    vision = False
    if vision_data_urls:
        for url in vision_data_urls:
            content_blocks.append({"type": "image_url",
                                   "image_url": {"url": url}})
        vision = True
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": content_blocks}]
    content, usage = chat_complete(
        cfg.model, messages,
        max_tokens=max_tokens or cfg.max_tokens,
        temperature=0, json_mode=True, timeout=cfg.timeout)
    tracker.record_call(usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0), vision=vision)
    return content


class Agent:
    def __init__(self, data, cfg=None, tracker=None):
        self.data = data
        self.cfg = cfg or _config_mod.LLMConfig()
        self.tracker = tracker or UsageTracker()

    # -- step 1: request understanding ------------------------------------
    def interpret_request(self, req, profile, trace):
        """LLM extraction validated against CSV record; deterministic fallback."""
        ctx = {"request_text": req.get("request_text", ""),
               "request_type": req.get("request_type", ""),
               "requested_amount": req.get("requested_amount", ""),
               "desired_completion_date": req.get("desired_completion_date", ""),
               "allows_partial_payment": req.get("allows_partial_payment", ""),
               "currency": profile.get("home_currency", "")}
        try:
            text = _call_model(
                self.cfg, self.tracker, INTERPRET_SYSTEM,
                INTERPRET_USER.format(**ctx))
            out = validate_interpretation(parse_json_object(text), {
                **req, "currency_hint": profile.get("home_currency", "")})
            trace["tools"].append("interpret_request(model)")
            _log(trace, "interpreted intent=%s via model" % out["intent"][:40])
            return out
        except (ModelError, ValueError) as e:
            _log(trace, "interpret_request fallback: %s" % type(e).__name__)
            return {"intent": req.get("request_type", ""),
                    "requested_amount": float(str(req["requested_amount"]).replace(",", "")),
                    "currency": profile.get("home_currency", ""),
                    "desired_completion_date": req["desired_completion_date"],
                    "payment_preference": "",
                    "constraints": []}

    # -- steps 4-5: conditional evidence gathering -------------------------
    def gather_evidence(self, req, ctx, interpretation, trace):
        """Message/image tools only when linked evidence exists."""
        extra_notes = []
        uid = req["user_id"]
        rid = req["request_id"]
        msg_info = tools.search_messages(self.data, uid, rid)
        trace["tools"].append("search_messages")
        _log(trace, "found %d messages" % len(msg_info["messages"]))
        if msg_info["messages"] and self.cfg.is_configured():
            known = [e["event_id"] for e in self.data.events_by_user.get(uid, [])][:60]
            digest = "\n".join(
                "[%s|%s] %s" % (m["message_id"], m["source_type"], m["message_text"][:400])
                for m in msg_info["messages"][:6])
            try:
                text = _call_model(
                    self.cfg, self.tracker, MESSAGE_SYSTEM,
                    MESSAGE_USER.format(messages=digest, event_ids=", ".join(known)))
                facts = parse_json_object(text).get("facts", [])
                valid = [f for f in facts if isinstance(f, dict)
                         and f.get("event_id") in {e["event_id"]
                                                   for e in self.data.events_by_user.get(uid, [])}]
                trace["tools"].append("interpret_messages(model)")
                _log(trace, "model extracted %d validated message facts" % len(valid))
                for f in valid[:4]:
                    extra_notes.append("message %s: %s %s" % (
                        f.get("event_id"), f.get("type"),
                        f.get("note", ""))[:120])
            except (ModelError, ValueError) as e:
                _log(trace, "message interpretation fallback: %s" % type(e).__name__)
        linked_images = [im for im in self.data.images
                         if im.get("request_id") == rid]
        for im in linked_images:
            info = tools.analyze_image(self.data, im)
            trace["tools"].append("analyze_image")
            if info.get("cached_amount"):
                extra_notes.append("image %s amount %s" % (
                    info["image_id"], info["cached_amount"]))
            if (self.cfg.is_configured() and self.cfg.vision_enabled
                    and not info["event_has_amount"]):
                path = self.data.base / "media" / "images" / (info["image_id"] + ".png")
                url = image_data_url(str(path))
                if url is not None:
                    try:
                        text = _call_model(
                            self.cfg, self.tracker,
                            "You read receipt amounts. Reply with ONLY JSON.",
                            '{"task":"read the total amount",'
                            ' "image_note":"receipt attached"}',
                            vision_data_urls=[url])
                        parsed = parse_json_object(text)
                        if "amount" in parsed:
                            extra_notes.append("vision amount %s" % str(parsed["amount"])[:40])
                        trace["tools"].append("analyze_image(vision)")
                    except (ModelError, ValueError) as e:
                        _log(trace, "vision fallback: %s" % type(e).__name__)
        return extra_notes

    # -- steps 6-10: deterministic evaluation -------------------------------
    def evaluate(self, ctx, trace):
        safe = tools.calculate_safe_amount(self.data, ctx)
        trace["tools"].append("calculate_safe_amount")
        earliest = tools.find_earliest_affordable_date(self.data, ctx)
        trace["tools"].append("find_earliest_affordable_date")
        std = tools.evaluate_payment_options(self.data, ctx)
        trace["tools"].append("evaluate_payment_options")
        raw = list(std["raw"])
        full_ok_today = any(c[5] == "full_payment" and c[8] == "affordable_now"
                            for c in raw)
        if not full_ok_today:
            flex = tools.evaluate_spending_changes(self.data, ctx)
            trace["tools"].append("evaluate_spending_changes")
            raw.extend(flex["raw"])
        ranked = tools.rank_strategies(raw, ctx["desired"])
        trace["tools"].append("rank_strategies")
        return safe, earliest, ranked, raw

    # -- steps 11-12: grounded explanation ----------------------------------
    def explain(self, req, ctx, winner, safe, earliest, notes, trace):
        """Model wording over tool facts only; template fallback otherwise."""
        plan = "; ".join("%s:%s" % (d, a) for d, a in winner["plan"]) if winner else "none"
        facts = {"user_id": req["user_id"],
                 "requested_amount": str(ctx["requested"]),
                 "currency": ctx["home"],
                 "request_type": req.get("request_type", ""),
                 "status": winner["status"] if winner else "not_affordable",
                 "method": winner["method"] if winner else "not_recommended",
                 "plan": plan,
                 "earliest": earliest["earliest_date_for_full_payment"] or "none",
                 "spending": self._spending_str(winner),
                 "safe": safe["amount_safe_to_pay"],
                 "minimum": str(ctx["minimum"]),
                 "context": "; ".join(notes[:6]) or "forecast and preferences"}
        allowed = set()
        for key in ("requested_amount", "safe", "minimum"):
            allowed |= numbers_in_text(str(facts[key]))
        allowed |= numbers_in_text(plan)
        allowed |= numbers_in_text(facts["earliest"])
        allowed |= numbers_in_text(facts["spending"])
        try:
            text = _call_model(self.cfg, self.tracker, EXPLAIN_SYSTEM,
                               EXPLAIN_USER.format(**facts))
            expl = parse_json_object(text).get("decision_explanation", "")
            if not isinstance(expl, str) or not explanation_is_grounded(expl, allowed):
                raise ValueError("ungrounded explanation")
            trace["tools"].append("explain(model)")
            return expl.strip()[:600]
        except (ModelError, ValueError):
            trace["tools"].append("explain(fallback)")
            return self._deterministic_explanation(ctx, winner, safe, earliest)

    @staticmethod
    def _deterministic_explanation(ctx, winner, safe, earliest):
        """Fallback wording identical to the deterministic engine."""
        from decimal import Decimal as _D
        if winner is None:
            plan, method, spending = [], "not_recommended", "none"
        else:
            plan = [(parse_date(d), _D(str(a))) for d, a in winner["plan"]]
            method, spending = winner["method"], Agent._spending_str(winner)
        early = (parse_date(earliest["earliest_date_for_full_payment"])
                 if earliest["earliest_date_for_full_payment"] else None)
        return _main.make_explanation(
            ctx["home"], ctx["start_balance"], ctx["minimum"],
            _D(str(safe["amount_safe_to_pay"])), ctx["requested"],
            method, plan, early, spending, ctx["forecast"])

    @staticmethod
    def _spending_str(winner):
        if not winner:
            return "none"
        parts = ["stop:%s" % e for e in sorted(winner.get("spending_stops", []))]
        parts += ["reduce_to:%s:%s" % (k, v)
                  for k, v in sorted(winner.get("spending_reductions", {}).items())]
        return "|".join(parts) or "none"

    # -- full pipeline for one request --------------------------------------
    def run_request(self, req):
        trace = _trace()
        model_available = self.cfg.is_configured()
        if not model_available:
            # Whole request served by the deterministic path; counted once.
            self.tracker.record_fallback()
            trace["fallbacks"] += 1
        try:
            profile = tools.get_financial_profile(self.data, req["user_id"])
            trace["tools"].append("get_financial_profile")
            interpretation = self.interpret_request(req, profile, trace)
            ctx = _main.prepare_request(self.data, req)
            trace["tools"].append("forecast_balance(via context)")
            notes = self.gather_evidence(req, ctx, interpretation, trace)
            safe, earliest, ranked, raw = self.evaluate(ctx, trace)
            hist = tools.get_financial_history(
                self.data, req["user_id"], req["request_date"], ctx["home"])
            trace["tools"].append("get_financial_history")
            pend = tools.get_pending_and_future_commitments(self.data, ctx)
            trace["tools"].append("get_pending_and_future_commitments")
            notes.append("safe today %s; earliest full %s; %d future flows" % (
                safe["amount_safe_to_pay"],
                earliest["earliest_date_for_full_payment"] or "never",
                pend["count"]))
            notes.append("past: %d income, %d expenses" % (
                hist["income_count"], hist["expense_count"]))
            notes.append("considers %s; protects %s" % (
                profile["payment_methods_user_will_consider"] or "nothing",
                profile["expense_categories_to_protect"] or "nothing"))
            winner = ranked["winner"]
            if winner is None:
                expl = self.explain(req, ctx, None, safe, earliest, notes, trace)
                row = _main.out_row(req["request_id"], Decimal(safe["amount_safe_to_pay"]),
                                    ctx["requested"], "not_affordable", "not_recommended",
                                    [], None, "none", expl)
                self._final_check(row)
                return row, trace
            # override protection: agent preference must match deterministic rank
            expl = self.explain(req, ctx, winner, safe, earliest, notes, trace)
            try:
                validate_final_result({"affordability_status": winner["status"],
                                       "recommended_payment_method": winner["method"]})
            except ValueError:
                trace["overrides"] += 1
                row = _main.decide_one(self.data, req)
                self.tracker.record_fallback()
                trace["fallbacks"] += 1
                return row, trace
            import datetime as _dt
            raw_winner = ranked["winner_raw"]
            plan = [(_dt.date.fromisoformat(d), Decimal(a))
                    for d, a in winner["plan"]]
            row = _main.out_row(
                req["request_id"], Decimal(safe["amount_safe_to_pay"]), ctx["requested"],
                winner["status"], winner["method"], plan,
                (parse_date(earliest["earliest_date_for_full_payment"])
                 if earliest["earliest_date_for_full_payment"] else None),
                self._spending_str(winner), expl)
            _ = raw_winner  # deterministic artifact retained for audit
            self._final_check(row)
            return row, trace
        except Exception as e:
            trace["fallbacks"] += 1
            self.tracker.record_fallback()
            _log(trace, "pipeline fallback: %s" % type(e).__name__)
            row = _main.decide_one(self.data, req)
            return row, trace

    @staticmethod
    def _final_check(row):
        assert row["affordability_status"] in (
            "affordable_now", "affordable_with_plan", "affordable_later",
            "not_affordable")
        assert row["recommended_payment_method"] in (
            "full_payment", "partial_payment", "installments", "wait",
            "not_recommended")
        assert row["decision_explanation"] and row["decision_explanation"].strip()
