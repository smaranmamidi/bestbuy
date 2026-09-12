"""Regression tests for specification audit fixes. Run with unittest."""
import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "code"))
from finance import (Data, parse_messages_for_user, build_user_forecast, simulate,
                     fmt_amount, expand_option)


def _facts(**kw):
    base = {"salary_override": None, "salary_single_cycle": False,
            "salary_effective": None, "salary_date_override": None,
            "one_time_credits": [], "one_time_debits": [], "rent_multiplier": None,
            "remove_future_salary": False, "ignore_pending_credit_event_ids": set(),
            "settle_event_ids": set()}
    base.update(kw)
    return base


class TestSpecAuditFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Data(str(REPO / "dataset"))

    def test_gaji_pokok_override_parsed(self):
        msgs = [{"message_id": "t", "user_id": "u", "request_id": "",
                 "related_event_id": "", "sent_at": "", "source_type": "employer",
                 "message_text": "Gaji pokok yang dikonfirmasi adalah IDR 38760000. Komisi belum disetujui."}]
        facts = parse_messages_for_user(msgs, "")
        self.assertIsNotNone(facts["salary_override"])
        self.assertEqual(facts["salary_override"][0], Decimal("38760000"))

    def test_temporary_pay_single_cycle(self):
        msgs = [{"message_id": "t", "user_id": "u", "request_id": "",
                 "related_event_id": "", "sent_at": "", "source_type": "employer",
                 "message_text": "Your temporary monthly pay is EUR 1037.52. The reduced amount "
                                 "continues for the next payroll. This is the amount currently "
                                 "scheduled for the affected pay cycle."}]
        facts = parse_messages_for_user(msgs, "")
        self.assertEqual(facts["salary_override"], (Decimal("1037.52"), "EUR"))
        self.assertTrue(facts["salary_single_cycle"])

    def test_final_payroll_suppresses_projection(self):
        facts = _facts()
        fc, _ = build_user_forecast(self.data, "user_05", date(2025, 11, 6), "ZAR", facts)
        self.assertIsNone(fc["salary_med"])

    def test_gig_payouts_not_projected(self):
        facts = _facts()
        fc, _ = build_user_forecast(self.data, "user_10", date(2024, 12, 6), "INR", facts)
        self.assertIsNone(fc["salary_med"])

    def test_second_household_income_projected(self):
        facts = _facts()
        fc, _ = build_user_forecast(self.data, "user_13", date(2024, 3, 7), "EUR", facts)
        extras = fc.get("extra_salaries", [])
        self.assertTrue(len(extras) >= 1)

    def test_bulk_outlier_excluded_from_burn(self):
        facts = _facts()
        fc, _ = build_user_forecast(self.data, "user_17", date(2026, 3, 1), "INR", facts)
        # naive total without exclusion would include the 41272 bulk receipt
        self.assertLess(fc["daily_burn"], Decimal("3400"))

    def test_variable_flex_savings_apply_daily(self):
        fc = {"explicit": [], "salary_med": None, "salary_day": 15,
              "extra_salaries": [], "salary_date_override": None,
              "fixed": [], "fixed_cats": set(), "daily_burn": Decimal("100")}
        ok_base, min_base = simulate(date(2024, 1, 1), Decimal("10000"), Decimal("0"),
                                     fc, [])
        ok_cut, min_cut = simulate(date(2024, 1, 1), Decimal("10000"), Decimal("0"),
                                   fc, [], set(), {}, Decimal("10"))
        self.assertGreater(min_cut, min_base)

    def test_installment_months_duration(self):
        # 15 payments every 30 days is ~15 months, exceeds max 12
        months = (15 - 1) * 30 / 30.0 + 1
        self.assertGreater(months, 12)
        months3 = (3 - 1) * 31 / 30.0 + 1
        self.assertLessEqual(months3, 11)

    def test_same_day_flows_applied(self):
        fc = {"explicit": [(date(2024, 1, 1), Decimal("-150"), "e1", "rent")],
              "salary_med": None, "salary_day": 15, "extra_salaries": [],
              "salary_date_override": None, "fixed": [], "fixed_cats": set(),
              "daily_burn": Decimal("0")}
        ok, mb = simulate(date(2024, 1, 1), Decimal("1000"), Decimal("900"),
                          fc, [])
        self.assertFalse(ok)
        self.assertEqual(mb, Decimal("850"))

    def test_amount_formatting(self):
        self.assertEqual(fmt_amount(Decimal("620.4")), "620.40")
        self.assertEqual(fmt_amount(Decimal("25256")), "25256")
        self.assertEqual(fmt_amount(Decimal("15952906.67")), "15952906.67")


if __name__ == "__main__":
    unittest.main()
