#!/usr/bin/env python3
"""Interactive frontend: pick or type a request, get Buy-or-Wait prediction.

Stdlib only. Run from repo root:
    python3 code/app.py [PORT]
Then open http://localhost:8000 in a browser.

Modes:
1. existing dataset request_id (full payment options + messages + LLM)
2. custom request for an existing dataset user_id (same engine, no options)
3. brand-new customer (no dataset history: typed balance/income/expenses)

GET  /                HTML form + result card
GET  /api/users       [{user_id, home_currency, ...}] (no secrets)
GET  /api/requests    [{request_id, user_id, requested_amount, ...}]
POST /api/predict     {mode: existing|custom|new_customer, ...} -> row JSON
"""
import html
import json
import sys
import urllib.parse
from datetime import date
from decimal import Decimal, InvalidOperation
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import main as _main
from finance import (Data, earliest_full_date, fmt_amount, max_safe_today,
                     parse_date)
from agent.config import LLMConfig
from agent.orchestrator import Agent
from agent.usage import UsageTracker

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET = REPO_ROOT / "dataset"

data = Data(str(DATASET))
cfg = LLMConfig(str(REPO_ROOT))
tracker = UsageTracker()
agent = Agent(data, cfg, tracker)


def _profile_summary(uid):
    p = data.profiles.get(uid, {})
    return {"user_id": uid,
            "home_currency": p.get("home_currency", ""),
            "current_available_balance": p.get("current_available_balance", ""),
            "minimum_balance_to_keep": p.get("minimum_balance_to_keep", ""),
            "payment_methods_user_will_consider":
                p.get("payment_methods_user_will_consider", "")}


def predict_existing(request_id):
    req = data.requests_by_id.get(request_id)
    if not req:
        raise ValueError("unknown request_id")
    row, _trace = agent.run_request(dict(req))
    return row


def predict_custom(user_id, requested_amount, request_date, desired_date,
                   request_type="purchase", allows_partial=True,
                   request_text=""):
    if user_id not in data.profiles:
        raise ValueError("unknown user_id")
    try:
        float(str(requested_amount).replace(",", ""))
    except ValueError:
        raise ValueError("requested_amount must be a number")
    req = {"request_id": "custom-preview",
           "user_id": user_id,
           "request_date": request_date or date.today().isoformat(),
           "request_type": request_type or "purchase",
           "requested_amount": str(requested_amount).replace(",", ""),
           "desired_completion_date":
               desired_date or request_date or date.today().isoformat(),
           "allows_partial_payment": "true" if allows_partial else "false",
           "request_text": request_text or
           f"Custom {request_type} request of {requested_amount}."}
    row, _trace = agent.run_request(req)
    row["request_id"] = "custom-preview"
    return row


def _num(value, name, minimum=None):
    try:
        d = Decimal(str(value).replace(",", "").strip() or "NaN")
    except (InvalidOperation, ValueError, AttributeError):
        raise ValueError(f"{name} must be a number")
    if d.is_nan() or d.is_infinite():
        raise ValueError(f"{name} must be a number")
    if minimum is not None and d < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return d


def predict_new_customer(p):
    """Brand-new customer with no dataset history.

    p keys: home_currency, current_available_balance, minimum_balance_to_keep,
    monthly_income, salary_day, fixed_monthly_expenses, variable_monthly_expenses,
    consider_full_payment, consider_partial_payment, consider_wait,
    requested_amount, request_date, desired_completion_date, request_type,
    allows_partial_payment, request_text.
    """
    home = (p.get("home_currency", "") or "").strip().upper() or "INR"
    balance = _num(p.get("current_available_balance", ""), "current balance", Decimal("0"))
    minimum = _num(p.get("minimum_balance_to_keep", ""), "minimum balance", Decimal("0"))
    requested = _num(p.get("requested_amount", ""), "requested amount", Decimal("0"))
    if requested <= 0:
        raise ValueError("requested amount must be > 0")
    income = _num(p.get("monthly_income", 0) or 0, "monthly income", Decimal("0"))
    fixed_m = _num(p.get("fixed_monthly_expenses", 0) or 0, "fixed expenses", Decimal("0"))
    var_m = _num(p.get("variable_monthly_expenses", 0) or 0, "variable expenses", Decimal("0"))
    try:
        salary_day = int(p.get("salary_day", 1) or 1)
    except (TypeError, ValueError):
        raise ValueError("salary day must be 1-28")
    salary_day = min(max(salary_day, 1), 28)
    request_date = parse_date(p.get("request_date", "") or date.today().isoformat())
    desired = parse_date(p.get("desired_completion_date", "") or
                         (p.get("request_date", "") or date.today().isoformat()))
    if request_date is None:
        raise ValueError("request_date must be YYYY-MM-DD")
    if desired is None:
        raise ValueError("desired_completion_date must be YYYY-MM-DD")
    request_type = (p.get("request_type", "") or "purchase").strip() or "purchase"
    allows_partial = bool(p.get("allows_partial_payment", True))
    consider = set()
    if p.get("consider_full_payment", True):
        consider.add("full_payment")
    if allows_partial and p.get("consider_partial_payment", True):
        consider.add("partial_payment")
    if p.get("consider_wait", True):
        consider.add("wait")
    if not consider:
        raise ValueError("enable at least one payment method")
    fixed = []
    if fixed_m > 0:
        fixed.append({"category": "housing", "amount": fixed_m, "day": 5,
                      "sample_event": {"event_id": "new-customer-fixed"}})
    forecast = {"explicit": [],
                "salary_med": income if income > 0 else None,
                "salary_day": salary_day,
                "extra_salaries": [],
                "salary_first_override": None,
                "salary_date_override": None,
                "fixed": fixed,
                "fixed_cats": {f["category"] for f in fixed},
                "daily_burn": (var_m / Decimal("30") if var_m > 0 else Decimal("0")),
                "var_by_cat": {}}
    safe = max_safe_today(request_date, balance, minimum, forecast, requested)
    safe = max(Decimal("0"), min(safe, requested))
    earliest = earliest_full_date(request_date, balance, minimum, forecast, requested)
    ctx = {"request_date": request_date, "desired": desired, "requested": requested,
           "allows_partial": allows_partial, "home": home,
           "start_balance": balance, "minimum": minimum, "consider": consider,
           "max_inst": None, "forecast": forecast, "flex_templates": [],
           "safe": safe, "earliest": earliest, "options": []}
    candidates = _main.evaluate_standard_candidates(data, ctx)
    winner = _main.rank_candidates(candidates, desired)
    rtype = (request_type or "purchase")
    rtext = (p.get("request_text", "") or f"New customer {rtype} of {requested} {home}.")
    req_like = {"user_id": "new-customer", "request_type": rtype,
                "request_text": rtext}
    notes = [f"new customer, no dataset history; balance {fmt_amount(balance)} {home}, "
             f"minimum {fmt_amount(minimum)} {home}, income {fmt_amount(income)}/mo"]
    trace = {"steps": [], "tools": [], "fallbacks": 0, "overrides": 0}
    if winner is None:
        expl = agent.explain(
            req_like, {"home": home, "requested": requested, "minimum": minimum,
                       "start_balance": balance, "forecast": forecast},
            None, {"amount_safe_to_pay": fmt_amount(safe)},
            {"earliest_date_for_full_payment": earliest.isoformat() if earliest else ""},
            notes, trace)
        return _main.out_row("new-customer-preview", safe, requested,
                             "not_affordable", "not_recommended", [], None,
                             "none", expl)
    _tag, _total, _start, _npay, _optid, method, plan, _sids, _rd, status, _cr = winner
    wtools = {"plan": [(d.isoformat(), str(a)) for d, a in plan],
              "status": status, "method": method,
              "spending_stops": [], "spending_reductions": {}}
    expl = agent.explain(
        req_like, {"home": home, "requested": requested, "minimum": minimum,
                   "start_balance": balance, "forecast": forecast},
        wtools, {"amount_safe_to_pay": fmt_amount(safe)},
        {"earliest_date_for_full_payment": earliest.isoformat() if earliest else ""},
        notes, trace)
    return _main.out_row("new-customer-preview", safe, requested, status, method,
                         plan, earliest, "none", expl)


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Buy or Wait? — interactive</title>
<style>
body{font-family:system-ui,sans-serif;max-width:860px;margin:24px auto;padding:0 16px}
.card{border:1px solid #ccc;border-radius:10px;padding:16px;margin:12px 0}
label{display:block;margin:8px 0 2px}input,select,textarea{width:100%;padding:8px}
button{padding:10px 18px;margin-top:12px;cursor:pointer}
pre{white-space:pre-wrap;background:#f6f6f6;padding:12px;border-radius:8px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
</style></head><body>
<h1>Buy or Wait? — interactive prediction</h1>
<div id="meta" class="card">loading…</div>
<div class="card"><h2>1. Predict an existing dataset request</h2>
<label>request_id</label><select id="req"></select>
<button onclick="goExisting()">Predict</button></div>
<div class="card"><h2>2. Custom request for an existing user</h2>
<div class="row"><div><label>user_id</label><select id="user"></select></div>
<div><label>request_type</label><select id="rtype">
<option>purchase</option><option>family_transfer</option><option>travel</option>
<option>education</option><option>debt_repayment</option><option>investment</option>
<option>housing</option><option>emergency_expense</option><option>other</option>
</select></div></div>
<div class="row"><div><label>requested_amount</label>
<input id="amt" value="25000"></div>
<div><label>allows_partial_payment</label><select id="part">
<option value="true">true</option><option value="false">false</option></select></div></div>
<div class="row"><div><label>request_date (YYYY-MM-DD)</label>
<input id="rdate" value="__TODAY__"></div>
<div><label>desired_completion_date</label><input id="ddate" value="__TODAY__"></div></div>
<label>request_text</label><textarea id="rtext" rows="2">Can I afford this laptop?</textarea>
<button onclick="goCustom()">Predict custom</button></div>
<div class="card"><h2>3. Brand-new customer (no dataset history)</h2>
<div class="row"><div><label>home_currency</label><select id="ncur">
<option>INR</option><option>ZAR</option><option>IDR</option><option>USD</option><option>EUR</option>
</select></div>
<div><label>request_type</label><select id="nrtype">
<option>purchase</option><option>family_transfer</option><option>travel</option>
<option>education</option><option>debt_repayment</option><option>investment</option>
<option>housing</option><option>emergency_expense</option><option>other</option>
</select></div></div>
<div class="row"><div><label>current_available_balance</label>
<input id="nbal" value="100000"></div>
<div><label>minimum_balance_to_keep</label><input id="nmin" value="20000"></div></div>
<div class="row"><div><label>monthly_income (0 = none)</label>
<input id="ninc" value="50000"></div>
<div><label>salary_day (1-28)</label><input id="nsday" value="1"></div></div>
<div class="row"><div><label>fixed monthly expenses (0 = none)</label>
<input id="nfix" value="20000"></div>
<div><label>variable monthly expenses (0 = none)</label><input id="nvar" value="15000"></div></div>
<div class="row"><div><label>requested_amount</label><input id="namt" value="25000"></div>
<div><label>allows_partial_payment</label><select id="npart">
<option value="true">true</option><option value="false">false</option></select></div></div>
<div class="row"><div><label>request_date (YYYY-MM-DD)</label>
<input id="nrdate" value="__TODAY__"></div>
<div><label>desired_completion_date</label><input id="nddate" value="__TODAY__"></div></div>
<div class="row"><div><label>accept full_payment</label><select id="ncf">
<option value="true">true</option><option value="false">false</option></select></div>
<div><label>accept partial_payment</label><select id="ncp">
<option value="true">true</option><option value="false">false</option></select></div></div>
<div class="row"><div><label>accept wait</label><select id="ncw">
<option value="true">true</option><option value="false">false</option></select></div>
<div><label>&nbsp;</label><span style="color:#666">installments need seller options, so new customers use full / partial / wait</span></div></div>
<label>request_text</label><textarea id="nrtext" rows="2">Can I afford this laptop?</textarea>
<button onclick="goNew()">Predict new customer</button></div>
<div class="card"><h2>Result</h2><pre id="out">no prediction yet</pre></div>
<script>
async function j(u,o){const r=await fetch(u,o);if(!r.ok)throw new Error(await r.text());return r.json()}
function show(row){document.getElementById('out').textContent=JSON.stringify(row,null,2)}
async function init(){
 const m=await j('/api/meta');document.getElementById('meta').textContent=
  `provider=${m.provider} model=${m.model} configured=${m.configured} | dataset: ${m.num_requests} requests, ${m.num_users} users`;
 const reqs=await j('/api/requests');const rs=document.getElementById('req');
 reqs.forEach(r=>{const o=document.createElement('option');o.value=r.request_id;
  o.textContent=`${r.request_id} | ${r.user_id} | ${r.requested_amount} (${r.request_date})`;rs.appendChild(o)});
 const users=await j('/api/users');const us=document.getElementById('user');
 users.forEach(u=>{const o=document.createElement('option');o.value=u.user_id;
  o.textContent=`${u.user_id} | ${u.home_currency} | bal ${u.current_available_balance}`;us.appendChild(o)});
}
async function goExisting(){show({status:'working…'});
 try{show(await j('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({mode:'existing',request_id:document.getElementById('req').value})}))}
 catch(e){show({error:String(e)})}}
async function goCustom(){show({status:'working…'});
 try{show(await j('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({mode:'custom',user_id:document.getElementById('user').value,
   requested_amount:document.getElementById('amt').value,
   request_date:document.getElementById('rdate').value,
   desired_completion_date:document.getElementById('ddate').value,
   request_type:document.getElementById('rtype').value,
   allows_partial_payment:document.getElementById('part').value==='true',
   request_text:document.getElementById('rtext').value})}))}
 catch(e){show({error:String(e)})}}
async function goNew(){show({status:'working…'});
 const v=id=>document.getElementById(id).value;
 try{show(await j('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({mode:'new_customer',home_currency:v('ncur'),
   current_available_balance:v('nbal'),minimum_balance_to_keep:v('nmin'),
   monthly_income:v('ninc'),salary_day:v('nsday'),
   fixed_monthly_expenses:v('nfix'),variable_monthly_expenses:v('nvar'),
   requested_amount:v('namt'),request_date:v('nrdate'),
   desired_completion_date:v('nddate'),request_type:v('nrtype'),
   allows_partial_payment:v('npart')==='true',
   consider_full_payment:v('ncf')==='true',
   consider_partial_payment:v('ncp')==='true',
   consider_wait:v('ncw')==='true',
   request_text:v('nrtext')})}))}
 catch(e){show({error:String(e)})}}
init();
</script></body></html>
""".replace("__TODAY__", date.today().isoformat())


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json", code=200):
        raw = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/":
            self._send(PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/meta":
            d = cfg.describe()
            self._send(json.dumps({"provider": d["provider"],
                                   "model": d["model"],
                                   "configured": d["configured"],
                                   "num_requests": len(data.requests),
                                   "num_users": len(data.profiles)}))
        elif self.path == "/api/users":
            self._send(json.dumps([_profile_summary(u)
                                   for u in sorted(data.profiles)]))
        elif self.path == "/api/requests":
            self._send(json.dumps(
                [{"request_id": r["request_id"], "user_id": r["user_id"],
                  "request_date": r["request_date"],
                  "requested_amount": r["requested_amount"]}
                 for r in data.requests]))
        else:
            self._send(json.dumps({"error": "not found"}), code=404)

    def do_POST(self):
        if self.path != "/api/predict":
            self._send(json.dumps({"error": "not found"}), code=404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            if payload.get("mode") == "existing":
                row = predict_existing(payload.get("request_id", ""))
            elif payload.get("mode") == "new_customer":
                row = predict_new_customer(payload)
            else:
                row = predict_custom(
                    payload.get("user_id", ""),
                    payload.get("requested_amount", ""),
                    payload.get("request_date", ""),
                    payload.get("desired_completion_date", ""),
                    payload.get("request_type", "purchase"),
                    bool(payload.get("allows_partial_payment", True)),
                    payload.get("request_text", ""))
            self._send(json.dumps(row))
        except ValueError as e:
            self._send(json.dumps({"error": html.escape(str(e))}), code=400)
        except Exception:
            self._send(json.dumps({"error": "prediction failed"}), code=500)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    srv = HTTPServer(("127.0.0.1", port), H)
    d = cfg.describe()
    print(f"Buy or Wait? interactive on http://localhost:{port}")
    print(f"provider={d['provider']} model={d['model']} configured={d['configured']}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
