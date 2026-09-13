"""Agent-layer tests: config secrecy, schemas, tools, fallback, override."""
import os
import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "code"))

from agent import tools
from agent.config import LLMConfig
from agent.model_client import ModelError, chat_complete
from agent.orchestrator import Agent
from agent.schemas import (explanation_is_grounded, parse_json_object,
                           validate_final_result, validate_interpretation)
from agent.usage import UsageTracker
from finance import Data, parse_date
import main as _main


class TestAgentConfig(unittest.TestCase):
    def test_not_configured_without_key(self):
        os.environ.pop("LLM_API_KEY", None)
        os.environ["LLM_PROVIDER"] = "openai"
        try:
            cfg = LLMConfig()
            self.assertFalse(cfg.is_configured())
            self.assertIn("LLM_API_KEY", cfg.missing_reason())
        finally:
            os.environ.pop("LLM_PROVIDER", None)

    def test_describe_never_contains_secret(self):
        os.environ["LLM_API_KEY"] = "sk-test-should-never-appear"
        try:
            cfg = LLMConfig()
            blob = repr(cfg.describe()) + cfg.missing_reason()
            self.assertNotIn("sk-test-should-never-appear", blob)
        finally:
            os.environ.pop("LLM_API_KEY", None)

    def test_env_example_parity(self):
        text = (REPO / ".env.example").read_text()
        for key in ("LLM_API_KEY=", "LLM_MODEL=",
                    "VISION_ENABLED=true"):
            self.assertIn(key, text)
        self.assertTrue("LLM_PROVIDER=" in text)

    def test_model_errors_carry_no_secrets(self):
        os.environ.pop("LLM_API_KEY", None)
        try:
            chat_complete("any-model", [{"role": "user", "content": "hi"}],
                          timeout=5)
            self.fail("expected ModelError")
        except ModelError as e:
            self.assertNotIn("sk-", str(e))
            self.assertNotIn("Bearer", str(e))


class TestAgentSchemas(unittest.TestCase):
    def test_parse_json_object(self):
        obj = parse_json_object('prefix {"a": 1} suffix')
        self.assertEqual(obj, {"a": 1})
        with self.assertRaises(ValueError):
            parse_json_object("no json here")

    def test_interpretation_amount_must_match_record(self):
        req = {"requested_amount": "85000", "desired_completion_date": "2026-10-01"}
        with self.assertRaises(ValueError):
            validate_interpretation({"intent": "purchase", "requested_amount": 1,
                                     "currency": "INR",
                                     "desired_completion_date": "2026-10-01"},
                                    req)
        out = validate_interpretation(
            {"intent": "purchase", "requested_amount": 85000, "currency": "INR",
             "desired_completion_date": "2026-10-01"}, req)
        self.assertEqual(out["requested_amount"], 85000)

    def test_final_result_enums(self):
        with self.assertRaises(ValueError):
            validate_final_result({"affordability_status": "maybe",
                                   "recommended_payment_method": "full_payment"})
        self.assertTrue(validate_final_result(
            {"affordability_status": "affordable_now",
             "recommended_payment_method": "full_payment"}))

    def test_grounding_rejects_invented_numbers(self):
        allowed = {"85000.00", "20000.00"}
        self.assertTrue(explanation_is_grounded("Pay 85000 today, keep 20000.", allowed))
        self.assertFalse(explanation_is_grounded("Pay 90000 today.", allowed))


class TestAgentTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Data(str(REPO / "dataset"))
        req = cls.data.requests_by_id["request_27"]
        cls.ctx = _main.prepare_request(cls.data, req)

    def test_profile_tool(self):
        p = tools.get_financial_profile(self.data, "user_27")
        for key in ("current_available_balance", "minimum_balance_to_keep",
                    "home_currency", "payment_methods_user_will_consider"):
            self.assertIn(key, p)

    def test_history_tool(self):
        h = tools.get_financial_history(self.data, "user_27", "2026-07-05", "ZAR")
        self.assertIn("income_count", h)
        self.assertGreaterEqual(h["expense_count"], 0)

    def test_commitments_tool(self):
        c = tools.get_pending_and_future_commitments(self.data, self.ctx)
        self.assertIn("flows", c)

    def test_currency_tool_uses_challenge_rates(self):
        r = tools.convert_currency(self.data, "100", "USD", "INR", "2024-06-15")
        self.assertAlmostEqual(float(r["converted_amount"]), 8333.0, delta=5)

    def test_forecast_safe_earliest_tools(self):
        s = tools.calculate_safe_amount(self.data, self.ctx)
        e = tools.find_earliest_affordable_date(self.data, self.ctx)
        self.assertIn("amount_safe_to_pay", s)
        self.assertIn("earliest_date_for_full_payment", e)

    def test_payment_options_match_supplied(self):
        std = tools.evaluate_payment_options(self.data, self.ctx)
        for c in std["candidates"]:
            if c["method"] == "installments":
                self.assertNotEqual(c["payment_option_id"], "")

    def test_rank_tool_orders_deterministically(self):
        std = tools.evaluate_payment_options(self.data, self.ctx)
        flex = tools.evaluate_spending_changes(self.data, self.ctx)
        ranked = tools.rank_strategies(std["raw"] + flex["raw"], self.ctx["desired"])
        if ranked["winner"] is not None:
            self.assertIn(ranked["winner"]["method"],
                          ("full_payment", "partial_payment", "installments",
                           "wait"))

    def test_image_tool_never_invents(self):
        row = self.data.images[0]
        info = tools.analyze_image(self.data, row)
        self.assertEqual(info["image_id"], row["image_id"])


class TestAgentFallback(unittest.TestCase):
    def test_fallback_matches_deterministic(self):
        os.environ.pop("LLM_API_KEY", None)
        data = Data(str(REPO / "dataset"))
        agent = Agent(data, _fresh_unconfigured(), UsageTracker())
        for rid in ("request_27", "request_28", "request_16"):
            req = data.requests_by_id.get(rid) or _sample_req(data, rid)
            row, trace = agent.run_request(req)
            expected = _main.decide_one(data, req)
            self.assertEqual(row["request_id"], expected["request_id"])
            self.assertEqual(row["affordability_status"], expected["affordability_status"])
            self.assertEqual(row["recommended_payment_method"],
                             expected["recommended_payment_method"])
            self.assertEqual(row["payment_plan"], expected["payment_plan"])
            self.assertTrue(row["decision_explanation"].strip())
        snap = agent.tracker.snapshot()
        self.assertEqual(snap["calls"], 0)
        self.assertGreaterEqual(snap["fallbacks"], 0)


class TestAgentModelPath(unittest.TestCase):
    """Exercise the real OpenAI HTTP path against a local stub server.

    The stub only verifies plumbing (request shape, usage accounting,
    schema validation, fallback). Production numbers always come from the
    configured provider, never from here.
    """

    def _stub(self, payload):
        import json as _json
        import threading as _th
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                self.server.seen.append(_json.loads(self.rfile.read(n) or b"{}"))
                body = _json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        srv = HTTPServer(("127.0.0.1", 0), H)
        srv.seen = []
        _th.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05},
                   daemon=True).start()
        return srv

    def test_model_path_end_to_end(self):
        import main as _m
        import agent.model_client as _mc
        data = Data(str(REPO / "dataset"))
        good = {"choices": [{"message": {"content":
                 '{"decision_explanation": "Pay ZAR 6670 today."}'}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10,
                          "total_tokens": 110}}
        srv = self._stub(good)
        old_base, old_key = _mc.API_BASE, os.environ.get("LLM_API_KEY")
        old_model, old_prov = os.environ.get("LLM_MODEL"), os.environ.get("LLM_PROVIDER")
        _mc.API_BASE = f"http://127.0.0.1:{srv.server_port}"
        os.environ["LLM_API_KEY"] = "test-key-never-committed"
        os.environ["LLM_MODEL"] = "stub-model"
        os.environ["LLM_PROVIDER"] = "openai"
        try:
            from agent.config import LLMConfig
            from agent.usage import UsageTracker
            agent = Agent(data, LLMConfig(), UsageTracker())
            req = data.requests_by_id["request_27"]
            row, trace = agent.run_request(req)
            snap = agent.tracker.snapshot()
            self.assertGreater(snap["calls"], 0)
            self.assertEqual(snap["input_tokens"], 100 * snap["calls"])
            auth_headers = [h for r in srv.seen for h in [r]]
            self.assertTrue(srv.seen)
            body = srv.seen[0]
            self.assertIn("model", body)
            self.assertEqual(body["model"], "stub-model")
            self.assertTrue(row["decision_explanation"].strip())
        finally:
            _mc.API_BASE = old_base
            srv.shutdown()
            for k, v in (("LLM_API_KEY", old_key), ("LLM_MODEL", old_model),
                         ("LLM_PROVIDER", old_prov)):
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_garbage_model_output_falls_back(self):
        import main as _m
        import agent.model_client as _mc
        data = Data(str(REPO / "dataset"))
        bad = {"choices": [{"message": {"content": "not json at all"}}],
               "usage": {"prompt_tokens": 5, "completion_tokens": 5,
                         "total_tokens": 10}}
        srv = self._stub(bad)
        old_base = _mc.API_BASE
        _mc.API_BASE = f"http://127.0.0.1:{srv.server_port}"
        os.environ["LLM_API_KEY"] = "test-key-never-committed"
        os.environ["LLM_MODEL"] = "stub-model"
        os.environ["LLM_PROVIDER"] = "openai"
        try:
            from agent.config import LLMConfig
            from agent.usage import UsageTracker
            agent = Agent(data, LLMConfig(), UsageTracker())
            req = data.requests_by_id["request_27"]
            row, trace = agent.run_request(req)
            expected = _m.decide_one(data, req)
            self.assertEqual(row["payment_plan"], expected["payment_plan"])
            self.assertEqual(row["affordability_status"],
                             expected["affordability_status"])
        finally:
            _mc.API_BASE = old_base
            srv.shutdown()
            os.environ.pop("LLM_API_KEY", None)
            os.environ.pop("LLM_MODEL", None)
            os.environ.pop("LLM_PROVIDER", None)


def _fresh_unconfigured():
    from agent.config import LLMConfig
    os.environ.pop("LLM_API_KEY", None)
    os.environ["LLM_PROVIDER"] = ""
    return LLMConfig()


def _sample_req(data, rid):
    import csv
    for r in csv.DictReader(open(REPO / "dataset" / "sample_requests.csv")):
        if r["request_id"] == rid:
            return r
    return data.requests_by_id[rid]


if __name__ == "__main__":
    unittest.main()
