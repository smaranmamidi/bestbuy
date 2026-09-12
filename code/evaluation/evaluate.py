#!/usr/bin/env python3
"""Sample evaluation: compare engine output to dataset/sample_requests.csv."""
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from finance import Data
from main import decide_one
REPO = Path(__file__).resolve().parent.parent.parent
def num(a):
    try: return float((a or '').strip() or 0)
    except: return 0
def main():
    data = Data(str(REPO/"dataset"))
    samples = list(csv.DictReader(open(REPO/"dataset"/"sample_requests.csv")))
    keys = ["affordability_status","recommended_payment_method","payment_plan","earliest_date_for_full_payment","spending_changes_needed"]
    ok = {k:0 for k in keys}; safe_ok=0
    for s in samples:
        req = {k:s[k] for k in ["request_id","user_id","request_date","request_type","requested_amount","desired_completion_date","allows_partial_payment","request_text"]}
        pred = decide_one(data, req)
        for k in keys:
            if (pred[k] or "")==(s[k] or ""): ok[k]+=1
        if abs(num(pred["amount_safe_to_pay"])-num(s["amount_safe_to_pay"]))/max(1,num(s["amount_safe_to_pay"]))<0.05: safe_ok+=1
    print(f"samples: {len(samples)}")
    for k in keys: print(f" {k}: {ok[k]}/{len(samples)}")
    print(f" amount_safe_to_pay (5% tol): {safe_ok}/{len(samples)}")
if __name__ == "__main__":
    main()
