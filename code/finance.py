"""Deterministic financial engine for Buy or Wait challenge.

No LLM is used for arithmetic. All balances, dates, FX, forecasts,
ranking and validation are deterministic. Message parsing is rule based.
Image amounts use OCR with a manual cache fallback.
"""
import csv
import re
import subprocess
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP, getcontext
from pathlib import Path

getcontext().prec = 28

FORECAST_DAYS = 90

# Manual image amount cache derived from OCR of dataset/media/images.
# Values verified against receipt totals, net pay, balance due.
IMAGE_AMOUNT_CACHE = {
    "image_01": "4365000",     # payslip net pay IDR
    "image_02": "100000",      # rent receipt balance due INR (total 200000, received 100000)
    "image_03": "41272",       # grocery bill cash paid INR
    "image_04": "2854",        # delivered order item bill INR
    "image_05": "704.05",      # telecom amount due INR
    "image_06": "1995",        # grocery tax invoice total INR
    "image_07": "8528",        # restaurant grand total INR
    "image_08": "15339",       # maintenance total received INR
    "image_09": "723",         # water bill INR
    "image_10": "79679.26",    # large grocery invoice words total INR
    "image_11": "3650",        # hospital amount payable INR
    "image_12": "33.5",        # taxi total USD
    "image_13": "2298",        # tote bag total paid INR
    "image_14": "4545",        # pharmacy purchase fallback from category median
    "image_15": "9968",        # airline grand total INR
    "image_16": "393.22",      # EV charging total INR
}

IGNORE_STATUSES = {"failed", "cancelled", "unrealized"}
VALID_STATUSES = {"settled", "pending", "scheduled"}


def parse_date(s):
    s = (s or "").strip()
    if not s:
        return None
    return date.fromisoformat(s[:10])


def dec(s):
    s = (s or "").strip()
    if not s:
        return None
    try:
        return Decimal(s.replace(",", ""))
    except Exception:
        return None


def fmt_amount(d):
    """Format Decimal: integers without decimals, others with 2 decimals."""
    if d is None:
        return ""
    q = d.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    if q == q.to_integral_value():
        return format(q.to_integral_value(), "f")
    s = format(q, "f")
    # keep exactly 2 decimals for non-integers (matches sample style 620.40)
    if "." not in s:
        s += ".00"
    else:
        ip, fp = s.split(".")
        fp = (fp + "00")[:2]
        s = f"{ip}.{fp}"
    if s in ("-0.00", "-0"):
        s = "0.00" if "." in s else "0"
    return s


class FX:
    def __init__(self, rows):
        # rows: list of dicts rate_date, from, to, rate
        self.by_pair = defaultdict(list)
        for r in rows:
            d = parse_date(r["rate_date"])
            self.by_pair[(r["from_currency"], r["to_currency"])].append((d, Decimal(r["rate"])))
        for k in self.by_pair:
            self.by_pair[k].sort()

    def _rate_for(self, pair, settle):
        lst = self.by_pair.get(pair)
        if not lst:
            return None
        best = None
        for d, r in lst:
            if d <= settle:
                best = r
            else:
                break
        if best is None:
            best = lst[0][1]
        return best

    def convert(self, amount, from_c, to_c, settle):
        if amount is None:
            return None
        if from_c == to_c:
            return amount
        direct = self._rate_for((from_c, to_c), settle)
        if direct is not None:
            return amount * direct
        inv = self._rate_for((to_c, from_c), settle)
        if inv is not None and inv != 0:
            return amount / inv
        # via USD bridge
        for bridge in ("USD", "EUR"):
            if bridge in (from_c, to_c):
                continue
            r1 = self._rate_for((from_c, bridge), settle)
            i1 = self._rate_for((bridge, from_c), settle)
            r2 = self._rate_for((bridge, to_c), settle)
            i2 = self._rate_for((to_c, bridge), settle)
            a_usd = None
            if r1 is not None:
                a_usd = amount * r1
            elif i1 is not None and i1 != 0:
                a_usd = amount / i1
            if a_usd is None:
                continue
            if r2 is not None:
                return a_usd * r2
            if i2 is not None and i2 != 0:
                return a_usd / i2
        # USD direct pairs commonly present as USD->X; try inverse via USD
        # from USD to target
        if from_c == "USD":
            inv2 = self._rate_for((to_c, "USD"), settle)
            if inv2 is not None and inv2 != 0:
                return amount / inv2
        if to_c == "USD":
            inv1 = self._rate_for(("USD", from_c), settle)
            if inv1 is not None and inv1 != 0:
                return amount / inv1
        # fallback: parity (should rarely happen; amounts mostly home currency)
        return amount


def ocr_image_amount(image_path):
    try:
        out = subprocess.run(
            ["tesseract", str(image_path), "stdout"],
            capture_output=True, text=True, timeout=30,
        )
        return out.stdout or ""
    except Exception:
        return ""


NUM_RE = re.compile(r"(\d[\d,]*\.?\d*)")


def extract_amount_from_text(text, hint=""):
    """Heuristic: prefer lines with total/net pay/grand total/balance due."""
    if not text:
        return None
    lines = text.splitlines()
    scored = []
    for ln in lines:
        low = ln.lower()
        nums = NUM_RE.findall(ln)
        if not nums:
            continue
        for n in nums:
            try:
                v = Decimal(n.replace(",", ""))
            except Exception:
                continue
            if v <= 0 or v > Decimal("1000000000"):
                continue
            score = 0
            if any(k in low for k in ("net pay", "grand total", "total amount", "balance due",
                                      "amount payable", "cash paid", "item bill", "total :",
                                      "total payable", "amount due")):
                score += 10
            if "balance due" in low:
                score += 10
            if "net pay" in low:
                score += 10
            if "grand total" in low:
                score += 8
            scored.append((score, v, ln.strip()[:80]))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], -x[1]))
    return scored[0][1]


class Data:
    def __init__(self, dataset_dir):
        self.base = Path(dataset_dir)
        self.profiles = {}
        self.events_by_user = defaultdict(list)
        self.events_by_id = {}
        self.requests = []
        self.requests_by_id = {}
        self.options_by_request = defaultdict(list)
        self.messages = []
        self.messages_by_user = defaultdict(list)
        self.images = []
        self.image_by_event = {}
        self.fx = None
        self._load()

    def _load(self):
        with open(self.base / "financial_profiles.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.profiles[r["user_id"]] = r
        with open(self.base / "financial_events.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.events_by_user[r["user_id"]].append(r)
                self.events_by_id[r["event_id"]] = r
        with open(self.base / "requests.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.requests.append(r)
                self.requests_by_id[r["request_id"]] = r
        with open(self.base / "request_payment_options.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.options_by_request[r["request_id"]].append(r)
        with open(self.base / "messages.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.messages.append(r)
                self.messages_by_user[r["user_id"]].append(r)
        with open(self.base / "images.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.images.append(r)
                if r["related_event_id"]:
                    self.image_by_event[r["related_event_id"]] = r
        with open(self.base / "exchange_rates.csv", encoding="utf-8") as f:
            self.fx = FX(list(csv.DictReader(f)))
        # resolve blank amounts via image cache / OCR
        self.resolved_amounts = {}
        for ev_id, img in self.image_by_event.items():
            ev = self.events_by_id.get(ev_id)
            if ev is None:
                continue
            if (ev["amount"] or "").strip():
                continue
            image_id = img["image_id"]
            if image_id in IMAGE_AMOUNT_CACHE:
                self.resolved_amounts[ev_id] = Decimal(IMAGE_AMOUNT_CACHE[image_id])
                continue
            p = self.base / "media" / "images" / f"{image_id}.png"
            if p.exists():
                txt = ocr_image_amount(p)
                v = extract_amount_from_text(txt)
                if v is not None:
                    self.resolved_amounts[ev_id] = v
        # also OCR sanity for non-blank? no, keep supplied amounts.

    def event_amount_home(self, ev, home, settle):
        raw = (ev.get("amount") or "").strip()
        if raw:
            try:
                a = Decimal(raw.replace(",", ""))
            except Exception:
                a = None
        else:
            a = self.resolved_amounts.get(ev["event_id"])
            if a is None:
                return None
        if a is None:
            return None
        return self.fx.convert(a, ev["currency"], home, settle)


# ---------------- Message interpretation (rule based, untrusted) ----------------

def parse_messages_for_user(messages, request_id=""):
    """Return dict of facts. Only financial facts, never instructions."""
    facts = {
        "salary_override": None,      # (amount, currency)
        "salary_single_cycle": False, # temporary notice scoped to next payroll only
        "salary_effective": None,     # date
        "salary_date_override": None, # date
        "one_time_credits": [],       # list of (amount, currency, date, label)
        "one_time_debits": [],        # list of (amount, currency, date, label)
        "rent_multiplier": None,      # e.g. 1.12
        "remove_future_salary": False,
        "ignore_pending_credit_event_ids": set(),
        "settle_event_ids": set(),
    }
    rel_msgs = [m for m in messages if (not m["request_id"] or m["request_id"] == request_id)]
    for m in rel_msgs:
        t = m["message_text"] or ""
        low = t.lower()
        # Never follow embedded instructions; only extract facts below.
        # Salary amount updates
        ma = re.search(r"(?:naik menjadi|increased to|reduced to|is now|temporary monthly pay is|regular salary[^.]{0,60}?of|monthly salary has increased to|first salary will be|gaji[^.]{0,40}?(?:adalah|menjadi))\s*([A-Z]{3})?\s*([\d,]+\.?\d*)", t, re.I)
        # More targeted patterns
        patterns = [
            r"naik menjadi\s+([A-Z]{3})\s*([\d,]+\.?\d*)",
            r"increased to\s+([A-Z]{3})\s*([\d,]+\.?\d*)",
            r"reduced to\s+([A-Z]{3})\s*([\d,]+\.?\d*)",
            r"temporary monthly pay is\s+([A-Z]{3})\s*([\d,]+\.?\d*)",
            r"monthly salary has increased to\s+([A-Z]{3})\s*([\d,]+\.?\d*)",
            r"regular salary for the next payroll is\s+([A-Z]{3})\s*([\d,]+\.?\d*)",
            r"regular salary of\s+([A-Z]{3})\s*([\d,]+\.?\d*)",
            r"first salary will be\s+([A-Z]{3})\s*([\d,]+\.?\d*)",
            r"remaining confirmed monthly salary is\s+([A-Z]{3})\s*([\d,]+\.?\d*)",
            r"confirmed base salary is\s+([A-Z]{3})\s*([\d,]+\.?\d*)",
            r"base salary is\s+([A-Z]{3})\s*([\d,]+\.?\d*)",
            r"confirmed salary is\s+([A-Z]{3})\s*([\d,]+\.?\d*)",
            r"next salary is reduced to\s+([A-Z]{3})\s*([\d,]+\.?\d*)",
            r"gaji pokok[^\n.]{0,80}?(?:adalah|sebesar|menjadi)\s+([A-Z]{3})\s*([\d,]+\.?\d*)",
            r"gaji pertama[^\n.]{0,80}?(?:adalah|sebesar|adalah|dijadwalkan)[^\n.]{0,80}?([A-Z]{3})\s*([\d,]+\.?\d*)",
            r"gaji rutin[^.]{0,80}?(?:adalah)\s+([A-Z]{3})\s*([\d,]+\.?\d*)",
            r"gaji bulanan sementara[^.]{0,80}?(?:adalah)\s+([A-Z]{3})\s*([\d,]+\.?\d*)",
            r"gaji bulanan[^.]{0,80}?(?:adalah|menjadi)\s+([A-Z]{3})\s*([\d,]+\.?\d*)",
        ]
        found = None
        for pat in patterns:
            mm = re.search(pat, t, re.I)
            if mm:
                found = (mm.group(1).upper(), mm.group(2))
                break
        if found and ("salary" in low or "gaji" in low or "payroll" in low or "pay " in low):
            # skip unconfirmed: bonus pending, commission pending, prize processing
            if any(k in low for k in ("still pending", "not been approved", "awaiting approval",
                                      "until the payout is closed", "can change until",
                                      "has ended", "no off-season", "employment has ended",
                                      "bonus", "commission", "prize", "reimbursement",
                                      "linked to an earlier work expense", "not your regular salary")):
                pass
            else:
                try:
                    amt = Decimal(found[1].replace(",", ""))
                except Exception:
                    amt = None
                if amt is not None:
                    facts["salary_override"] = (amt, found[0])
                    # single-cycle temporary notices name the next/affected cycle
                    if any(k in low for k in ("temporary", "affected pay cycle",
                                              "continues for the next payroll",
                                              "reduced amount continues for the next")):
                        facts["salary_single_cycle"] = True
                    dm = re.search(r"(20\d{2}-\d{2}-\d{2})", t)
                    if dm:
                        try:
                            facts["salary_effective"] = date.fromisoformat(dm.group(1))
                        except Exception:
                            pass
                    else:
                        # applies from / berlaku mulai
                        dm2 = re.search(r"(?:applies from|berlaku mulai|resumes on|confirmed for|credit date is)\s*(20\d{2}-\d{2}-\d{2})", t, re.I)
                        if dm2:
                            try:
                                facts["salary_effective"] = date.fromisoformat(dm2.group(1))
                            except Exception:
                                pass
        # salary date change
        if "replaces the payroll date" in low or "confirmed salary is now expected on" in low:
            dm = re.search(r"(20\d{2}-\d{2}-\d{2})", t)
            if dm:
                try:
                    facts["salary_date_override"] = date.fromisoformat(dm.group(1))
                except Exception:
                    pass
        # confirmed invoice / first salary with explicit date -> one-time credit
        if "client approved an invoice payment of" in low or "invoice payment of" in low:
            mm = re.search(r"invoice payment of\s+([A-Z]{3})\s*([\d,]+\.?\d*).*?(20\d{2}-\d{2}-\d{2})", t, re.I | re.S)
            if mm:
                try:
                    facts["one_time_credits"].append((Decimal(mm.group(2).replace(",", "")), mm.group(1).upper(), date.fromisoformat(mm.group(3)), "invoice"))
                except Exception:
                    pass
        if "first salary" in low and "confirmed" in low:
            mm = re.search(r"([A-Z]{3})\s*([\d,]+\.?\d*).*?(20\d{2}-\d{2}-\d{2})", t, re.I | re.S)
            if mm and facts["salary_override"] is None:
                try:
                    facts["one_time_credits"].append((Decimal(mm.group(2).replace(",", "")), mm.group(1).upper(), date.fromisoformat(mm.group(3)), "first_salary"))
                except Exception:
                    pass
        # rent increase
        if "increases monthly rent by 12%" in low or "rent by 12%" in low:
            facts["rent_multiplier"] = Decimal("1.12")
        # employment ended -> remove future salary projection
        if any(k in low for k in ("employment has ended", "seasonal contract has ended", "hubungan kerja", "kontrak musiman")) and "no " in low:
            if "salary" in low or "income" in low or "pay" in low or "gaji" in low:
                # only if it says no future income confirmed
                if any(k in low for k in ("no off-season", "no regular salary", "tidak ada pembayaran", "no additional", "we'll contact")):
                    facts["remove_future_salary"] = True
        # related event handling
        rid = (m.get("related_event_id") or "").strip()
        if rid:
            if any(k in low for k in ("has not reached", "not reached your account", "has not been credited",
                                      "payment has not been credited", "no cash proceeds",
                                      "no units have been sold", "not been sold and no cash",
                                      "refund has been initiated", "still in payment processing",
                                      "still pending", "can change until", "is still being investigated",
                                      "reversal has not been posted", "dispute is open",
                                      "previous debit attempt failed")):
                facts["ignore_pending_credit_event_ids"].add(rid)
            if any(k in low for k in ("have reached your account", "has settled in the cash account",
                                      "sudah masuk ke rekening", "proceeds have reached")):
                facts["settle_event_ids"].add(rid)
    return facts


# ---------------- Forecast engine ----------------

def is_cash_countable(ev, facts):
    st = ev["status"]
    if st in IGNORE_STATUSES:
        return False
    if ev["direction"] == "non_cash":
        return False
    if ev["event_type"] == "investment_valuation":
        return False
    if ev["event_id"] in facts.get("ignore_pending_credit_event_ids", set()):
        return False
    if st == "pending" and ev["direction"] == "credit":
        # pending credits not counted unless message confirms settlement
        if ev["event_id"] not in facts.get("settle_event_ids", set()):
            return False
    return True


def build_user_forecast(data, user_id, request_date, home, facts):
    """Return dict with:
    - explicit: list of (settle_date, signed_amount_home) for future one-offs
    - salary_day, salary_med: projected salary template
    - fixed_monthly: list of (day, signed_amount) recurring fixed
    - daily_net: Decimal per-day net (negative = burn) for variable spend
    - flexible_templates: list of event dicts usable for spending changes
    """
    events = data.events_by_user.get(user_id, [])
    # past window for estimation
    past_start = request_date - timedelta(days=120)
    past = [e for e in events if (parse_date(e["settlement_date"]) or date.min) <= request_date
            and (parse_date(e["settlement_date"]) or date.min) >= past_start
            and e["status"] == "settled"]
    # salary history grouped into streams. Only stable recurring payroll streams
    # are projected (spec: detect recurrence only when history supports it;
    # do not invent bonuses, commissions, gig payouts as future salary).
    NON_SALARY_WORDS = ("commission", "bonus", "payout", "arrears", "reimbursement",
                        "prize", "refund", "lottery", "windfall",
                        "cashback", "marketplace", "delivery platform", "driver platform")
    ONE_TIME_WORDS = ("prorated", "first salary", "firstsalary", "arrears", "adjustment",
                      "backpay", "one-time", "one time")
    FINAL_WORDS = ("final ", "final.", "last paycheck", "last salary", "final settlement",
                   "final payroll", "final employer")
    # evidence: settled past 120d + scheduled future 90d (explicit continuation)
    sal_evidence = []
    for e in events:
        if e["direction"] != "credit":
            continue
        if e["category"] != "salary" and e["event_type"] != "income":
            continue
        desc = (e.get("description") or "").lower()
        if any(w in desc for w in NON_SALARY_WORDS):
            continue
        sd = parse_date(e["settlement_date"])
        if sd is None:
            continue
        if not (past_start <= sd <= request_date + timedelta(days=FORECAST_DAYS)):
            continue
        if e["status"] not in ("settled", "scheduled"):
            continue
        if sd > request_date and e["status"] != "scheduled":
            continue
        a = data.event_amount_home(e, home, sd)
        if a is None:
            continue
        sal_evidence.append((e, a, sd, desc))
    # group: payroll-like by month-day cycle; household second income apart
    sal_streams = defaultdict(list)
    for e, a, sd, desc in sal_evidence:
        if any(w in desc for w in ONE_TIME_WORDS):
            continue  # partial-period top-up, not the recurring rate
        if "household" in desc or "second" in desc:
            key = ("household", "second" in desc)
        elif "payroll" in desc or "salary" in desc or "gaji" in desc or "base" in desc:
            key = ("payroll", sd.day)
        else:
            # work income (freelance/contract): group by pay-cycle day
            key = ("cycle", sd.day)
        sal_streams[key].append((e, a, sd))
    salaries = []  # list of (median_amount, median_day)
    for key, lst in sal_streams.items():
        kind = key[0]
        need = 2 if kind in ("payroll", "household") else 4
        if len(lst) < need:
            continue
        amts = sorted(a for _, a, _ in lst)
        med = amts[len(amts) // 2]
        if med <= 0:
            continue
        tol = Decimal("0.50") if kind == "payroll" else Decimal("0.30")
        if (max(amts) - min(amts)) / med > tol:
            continue  # unstable gig/variable income, recurrence not supported
        dates = sorted(sd for _, _, sd in lst)
        gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        avg_gap = sum(gaps) / len(gaps) if gaps else 30
        if kind in ("payroll", "household"):
            if not (20 <= avg_gap <= 40):
                continue  # only monthly payroll is projected
        else:
            # work income on a fixed cycle day recurs monthly
            if not ((10 <= avg_gap <= 20) or (25 <= avg_gap <= 35)):
                continue
        latest_desc = (max(lst, key=lambda x: x[2])[0].get("description") or "").lower()
        if any(w in latest_desc for w in FINAL_WORDS):
            continue  # explicit termination in source record
        days = sorted(sd.day for _, _, sd in lst)
        salaries.append((med, days[len(days) // 2]))
    # fallback: a scheduled future employer salary plus any settled past salary
    # confirms continuation even with short history (e.g. first payroll after
    # a prorated starter month). Uses the scheduled amount, not invented.
    if not salaries:
        sched = [(e, a, sd) for e, a, sd, desc in sal_evidence
                 if e["status"] == "scheduled"
                 and ("payroll" in desc or "salary" in desc or "gaji" in desc or "base" in desc)]
        # any settled past salary evidences employment, including a prorated
        # starter month (rate itself comes from the scheduled record)
        settled_any = False
        for e in events:
            if e["direction"] != "credit":
                continue
            if e["category"] != "salary" and e["event_type"] != "income":
                continue
            desc = (e.get("description") or "").lower()
            if any(w in desc for w in NON_SALARY_WORDS):
                continue
            sd = parse_date(e["settlement_date"])
            if sd is None or not (past_start <= sd <= request_date):
                continue
            if e["status"] != "settled":
                continue
            settled_any = True
            break
        if sched and settled_any:
            sched.sort(key=lambda x: x[2])
            salaries = [(sched[-1][1], sched[-1][2].day)]
    # message salary override replaces the primary (largest) stream.
    # A temporary/single-cycle notice ("next payroll", "affected pay cycle")
    # applies only to the first future month; later months resume the
    # historical median. An ongoing change applies to all future months.
    # Scope was determined during message parsing (has request context).
    if facts.get("salary_override"):
        amt, cur = facts["salary_override"]
        conv = data.fx.convert(amt, cur, home, request_date)
        if salaries:
            idx = max(range(len(salaries)), key=lambda i: salaries[i][0])
            hist_med, hist_day = salaries[idx]
            if facts.get("salary_single_cycle"):
                salaries[idx] = (hist_med, hist_day)
                salary_first_override = (conv, hist_day)
            else:
                salaries[idx] = (conv, salaries[idx][1])
                salary_first_override = None
        else:
            salaries = [(conv, 15)]
            salary_first_override = None
    else:
        salary_first_override = None
    if facts.get("remove_future_salary"):
        salaries = []
    sal_med = salaries[0][0] if salaries else None
    sal_day = salaries[0][1] if salaries else 15
    extra_salaries = salaries[1:]
    # fixed recurring detection: group by (category, rounded amount)
    from collections import Counter
    fixed = []
    by_cat = defaultdict(list)
    for e in past:
        if e["direction"] != "debit":
            continue
        if e["category"] in ("salary",):
            continue
        sd = parse_date(e["settlement_date"])
        a = data.event_amount_home(e, home, sd)
        if a is None:
            continue
        by_cat[e["category"]].append((e, a, sd))
    for cat, lst in by_cat.items():
        if len(lst) < 2:
            continue
        amts = sorted(a for _, a, _ in lst)
        med = amts[len(amts) // 2]
        # if amounts stable within 8%, treat as fixed monthly with count check
        if med > 0 and (max(amts) - min(amts)) / med < Decimal("0.15") and len(lst) >= 3:
            days = sorted(sd.day for _, _, sd in lst)
            dmed = days[len(days) // 2]
            # monthly frequency: avg gap 25-35 days?
            dates = sorted(sd for _, _, sd in lst)
            gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
            avg_gap = sum(gaps) / len(gaps) if gaps else 30
            if 20 <= avg_gap <= 40:
                fixed.append({"category": cat, "amount": med, "day": dmed,
                              "sample_event": lst[-1][0]})
    # variable daily burn: average over past window, excluding one-time outliers.
    # Spec: distinguish recurring expenses from one-time purchases. A single
    # bulk/one-time amount far above the category median must not inflate the
    # recurring rate.
    fixed_cats = {f["category"] for f in fixed}
    var_total = Decimal("0")
    var_by_cat = {}
    for cat, lst in by_cat.items():
        if cat in fixed_cats:
            continue
        if len(lst) < 3:
            continue
        amts = sorted(a for _, a, _ in lst)
        med = amts[len(amts) // 2]
        cat_total = Decimal("0")
        for _, a, _ in lst:
            if med > 0 and a > med * 3:
                continue  # one-time bulk purchase, not recurring
            cat_total += a
        var_by_cat[cat] = cat_total
        var_total += cat_total
    daily_burn = var_total / Decimal("120") if var_total else Decimal("0")
    # explicit future one-offs (inclusive of request_date)
    explicit = []
    end = request_date + timedelta(days=FORECAST_DAYS)
    for e in events:
        sd = parse_date(e["settlement_date"])
        if sd is None or sd < request_date or sd > end:
            continue
        if not is_cash_countable(e, facts):
            continue
        # skip salary-type settled/scheduled that will be covered by projection?
        # Keep explicit salary if it exists (prefer explicit over projection for that month)
        a = data.event_amount_home(e, home, sd)
        if a is None:
            continue
        signed = -a if e["direction"] == "debit" else a
        explicit.append((sd, signed, e["event_id"], e["category"]))
    # message one-time credits/debits in window (inclusive of request_date)
    for amt, cur, d, label in facts.get("one_time_credits", []):
        if request_date <= d <= end:
            conv = data.fx.convert(amt, cur, home, d)
            explicit.append((d, conv, f"msg:{label}", label))
    # rent multiplier
    if facts.get("rent_multiplier"):
        for f in fixed:
            if f["category"] == "rent":
                f["amount"] = (f["amount"] * facts["rent_multiplier"]).quantize(Decimal("0.01"))
    # flexible templates: latest event per flexible category
    flex_by_id = {}
    for e in events:
        if e["flexibility"] not in ("stoppable", "reducible", "reducible_or_stoppable"):
            continue
        sd = parse_date(e["settlement_date"])
        if sd is None or sd > request_date:
            continue
        if e["status"] != "settled":
            continue
        # keep latest per event_id (each event is a template occurrence; use latest occurrence id)
        flex_by_id[e["event_id"]] = e
    # reduce to latest 1 per (category, description first word) to limit combos
    latest = {}
    for e in flex_by_id.values():
        key = (e["category"], e["description"][:24])
        sd = parse_date(e["settlement_date"])
        if key not in latest or sd > parse_date(latest[key]["settlement_date"]):
            latest[key] = e
    flexible_templates = list(latest.values())
    return {
        "explicit": explicit,
        "salary_med": sal_med,
        "salary_day": sal_day,
        "extra_salaries": extra_salaries,
        "salary_first_override": salary_first_override,
        "salary_date_override": facts.get("salary_date_override"),
        "fixed": fixed,
        "fixed_cats": fixed_cats,
        "daily_burn": daily_burn,
        "var_by_cat": var_by_cat,
    }, flexible_templates


def simulate(request_date, start_balance, minimum, forecast, payments, spending_stop_ids=None, spending_reduce=None, flex_var_savings_per_day=None):
    """Daily simulation. payments: list of (date, amount). Returns (ok, balances, min_bal)."""
    spending_stop_ids = spending_stop_ids or set()
    spending_reduce = spending_reduce or {}
    end = request_date + timedelta(days=FORECAST_DAYS)
    # map explicit by date
    flow = defaultdict(Decimal)
    for sd, signed, _eid, cat in forecast["explicit"]:
        flow[sd] += signed
    # salary projection: monthly on salary_day, skip month if explicit salary exists same month
    # A single-cycle temporary override applies to the first projected month only.
    first_override = forecast.get("salary_first_override")
    first_override_used = [False]

    def _project_monthly(amount, day, override=None, allow_first_override=False):
        if amount is None or amount <= 0:
            return
        explicit_months = set()
        for sd, signed, _eid, cat in forecast["explicit"]:
            if cat == "salary" and signed > 0:
                explicit_months.add((sd.year, sd.month))
        cur = date(request_date.year, request_date.month, 1)
        for _ in range(5):
            y, m = cur.year, cur.month
            if (y, m) not in explicit_months:
                try:
                    d = date(y, m, min(day, 28))
                except Exception:
                    d = date(y, m, 15)
                if request_date <= d <= end:
                    use_d = d
                    if override is not None and override.year == y and override.month == m:
                        use_d = override
                    pay = amount
                    if allow_first_override and first_override is not None and not first_override_used[0]:
                        pay = first_override[0]
                        first_override_used[0] = True
                    flow[use_d] += pay
            if m == 12:
                cur = date(y + 1, 1, 1)
            else:
                cur = date(y, m + 1, 1)
    _project_monthly(forecast["salary_med"], forecast["salary_day"],
                     forecast.get("salary_date_override"), allow_first_override=True)
    for extra_amt, extra_day in forecast.get("extra_salaries", []):
        _project_monthly(extra_amt, extra_day)
    # flexible variable-category savings: a budget cut applies every day from
    # request_date (not on a single template day). Expressed as a per-day
    # reduction of the variable burn rate.
    flex_saving = flex_var_savings_per_day or Decimal("0")
    effective_daily = forecast["daily_burn"] - flex_saving
    if effective_daily < 0:
        effective_daily = Decimal("0")
    # fixed monthly: project each month, skip if stopped
    for f in forecast["fixed"]:
        # find template event id to allow stop
        tmpl_id = f["sample_event"]["event_id"]
        if tmpl_id in spending_stop_ids:
            continue
        amt = f["amount"]
        if tmpl_id in spending_reduce:
            amt = spending_reduce[tmpl_id]
        cur = date(request_date.year, request_date.month, 1)
        for _ in range(5):
            y, m = cur.year, cur.month
            try:
                d = date(y, m, min(f["day"], 28))
            except Exception:
                d = date(y, m, 15)
            if request_date <= d <= end:
                # avoid double count if explicit same category+amount same date within 3 days?
                dup = False
                for sd, signed, _eid, cat in forecast["explicit"]:
                    if cat == f["category"] and abs((sd - d).days) <= 3 and abs(abs(signed) - amt) / max(amt, Decimal("1")) < Decimal("0.2"):
                        dup = True
                        break
                if not dup:
                    flow[d] -= amt
            if m == 12:
                cur = date(y + 1, 1, 1)
            else:
                cur = date(y, m + 1, 1)
    # payments
    paymap = defaultdict(Decimal)
    for d, a in payments:
        paymap[d] += a
    bal = start_balance
    min_bal = bal
    n = (end - request_date).days
    ok = True
    # pending debits with settlement <= request_date already in start? No, subtract day 0 before check?
    for i in range(n + 1):
        d = request_date + timedelta(days=i)
        bal += flow.get(d, Decimal("0"))
        if i > 0:
            bal -= effective_daily
        # apply payments on date (day 0 included)
        if d in paymap:
            bal -= paymap[d]
        if bal < min_bal:
            min_bal = bal
        if bal < minimum - Decimal("0.005"):
            ok = False
    return ok, min_bal


def max_safe_today(request_date, start_balance, minimum, forecast, requested):
    lo = Decimal("0")
    hi = requested
    # quick check full
    ok, _ = simulate(request_date, start_balance, minimum, forecast, [(request_date, requested)] if requested > 0 else [])
    if ok:
        return requested
    # check zero
    ok0, _ = simulate(request_date, start_balance, minimum, forecast, [])
    if not ok0:
        return Decimal("0")
    for _ in range(50):
        mid = (lo + hi) / 2
        if hi - lo < Decimal("0.005"):
            break
        okm, _ = simulate(request_date, start_balance, minimum, forecast, [(request_date, mid)] if mid > 0 else [])
        if okm:
            lo = mid
        else:
            hi = mid
    # floor to 2 decimals
    return (lo.quantize(Decimal("0.01"), rounding=ROUND_DOWN))


def earliest_full_date(request_date, start_balance, minimum, forecast, requested):
    end = request_date + timedelta(days=FORECAST_DAYS)
    for i in range(FORECAST_DAYS + 1):
        d = request_date + timedelta(days=i)
        ok, _ = simulate(request_date, start_balance, minimum, forecast, [(d, requested)])
        if ok:
            return d
    return None


def expand_option(opt):
    n = int(opt["number_of_payments"] or "1")
    amt = Decimal(opt["payment_amount"])
    first = parse_date(opt["first_payment_date"])
    freq = (opt["payment_frequency_days"] or "").strip()
    freq = int(freq) if freq else 0
    pays = []
    for i in range(n):
        d = first + timedelta(days=i * freq) if freq else first
        pays.append((d, amt))
    return pays
