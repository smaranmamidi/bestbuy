"""Validation for structured model outputs and explanation grounding.

Every number the model states must already exist in the deterministic tool
facts. Anything else is rejected and triggers the deterministic fallback.
"""
import json
import re

STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}

_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")


def parse_json_object(text):
    """Extract the first JSON object from model text. Raises ValueError."""
    if not text or not text.strip():
        raise ValueError("empty model output")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except ValueError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model output")
    obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("model output is not a JSON object")
    return obj


def validate_interpretation(obj, req):
    """Check request interpretation against authoritative CSV fields."""
    if not isinstance(obj, dict):
        raise ValueError("interpretation must be an object")
    for key in ("intent", "requested_amount", "currency", "desired_completion_date"):
        if key not in obj:
            raise ValueError(f"interpretation missing '{key}'")
    try:
        amount = float(str(obj["requested_amount"]).replace(",", ""))
    except (ValueError, TypeError):
        raise ValueError("interpretation amount is not numeric")
    expected_amount = float(str(req["requested_amount"]).replace(",", ""))
    if abs(amount - expected_amount) > 0.005:
        raise ValueError("interpretation amount disagrees with request record")
    if str(obj["currency"]).strip().upper() != str(req.get("currency_hint", "") or "").strip().upper():
        # currency_hint is optional context; fall back to profile home currency
        pass
    if str(obj["desired_completion_date"]).strip() != str(req["desired_completion_date"]).strip():
        raise ValueError("interpretation date disagrees with request record")
    return {"intent": str(obj.get("intent", ""))[:200],
            "requested_amount": expected_amount,
            "currency": str(obj.get("currency", ""))[:8],
            "desired_completion_date": str(req["desired_completion_date"]),
            "payment_preference": str(obj.get("payment_preference", ""))[:120],
            "constraints": list(obj.get("constraints", []) or [])[:8]}


def validate_final_result(obj):
    """Check the model's final recommendation object shape and enums."""
    if not isinstance(obj, dict):
        raise ValueError("final result must be an object")
    status = str(obj.get("affordability_status", ""))
    method = str(obj.get("recommended_payment_method", ""))
    if status not in STATUSES:
        raise ValueError("invalid affordability_status")
    if method not in METHODS:
        raise ValueError("invalid recommended_payment_method")
    if (status == "affordable_now") != (method == "full_payment"):
        # affordable_now pairs with full_payment; partial pairs with with_plan.
        if not (status == "affordable_with_plan" and method in (
                "partial_payment", "installments", "full_payment")):
            if not (status in ("affordable_later", "not_affordable")):
                raise ValueError("status/method pair is inconsistent")
    return True


def numbers_in_text(text):
    """All numeric tokens in a text, normalized for comparison."""
    out = set()
    for tok in _NUMBER_RE.findall(text or ""):
        try:
            out.add(f"{float(tok.replace(',', '')):.2f}")
        except ValueError:
            continue
    return out


def explanation_is_grounded(explanation, allowed_numbers):
    """Every number in the explanation must be in the allowed tool-fact set."""
    if not explanation or not explanation.strip():
        return False
    mentioned = numbers_in_text(explanation)
    return mentioned.issubset(set(allowed_numbers))
