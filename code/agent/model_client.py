"""Multi-provider chat client over stdlib urllib. No third-party packages required.

Supported providers (via LLM_PROVIDER):
- openai: POST https://api.openai.com/v1/chat/completions, Bearer auth.
- gemini: POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent, key auth.

Security rules enforced here:
- The API key is read from the `LLM_API_KEY` environment variable at call
  time and is never stored on objects, never logged, and never included in
  exceptions, usage records, or reports.
- Only HTTPS is used (except localhost stub servers in tests). Image bytes
  are sent to the configured provider only.
"""
import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

API_BASE = "https://api.openai.com/v1"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class ModelError(Exception):
    """Model call failed. Message is always secret-free."""


def _api_key():
    return (os.environ.get("LLM_API_KEY", "") or "").strip()


def _provider():
    return (os.environ.get("LLM_PROVIDER", "") or "").strip().lower()


def _post_json(url, payload, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + _api_key()},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        try:
            detail = e.read()[:500].decode("utf-8", "replace")
        except Exception:
            detail = ""
        # Never include request bodies or headers (which carry the key).
        raise ModelError(f"provider HTTP {e.code}") from None
    except TimeoutError:
        raise ModelError("provider timeout") from None
    except OSError as e:
        raise ModelError(f"provider connection failed: {type(e).__name__}") from None


def _post_json_gemini(url, payload, timeout):
    """POST to Gemini. Key travels in the query string; never logged."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        # Do not include URL (carries key) or body in the error.
        raise ModelError(f"provider HTTP {e.code}") from None
    except TimeoutError:
        raise ModelError("provider timeout") from None
    except OSError as e:
        raise ModelError(f"provider connection failed: {type(e).__name__}") from None


def _blocks_to_text_and_images(content):
    """Split OpenAI-style content into (text, [(mime, b64), ...])."""
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    texts = []
    images = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                if isinstance(block, str):
                    texts.append(block)
                continue
            btype = block.get("type", "")
            if btype == "text":
                texts.append(str(block.get("text", "")))
            elif btype == "image_url":
                iu = block.get("image_url", {})
                url = iu.get("url", "") if isinstance(iu, dict) else str(iu)
                mime, b64 = _parse_data_url(url)
                if b64:
                    images.append((mime, b64))
            elif "text" in block:
                texts.append(str(block.get("text", "")))
    return "\n".join([t for t in texts if t]), images


def _parse_data_url(url):
    if not isinstance(url, str) or not url.startswith("data:"):
        return None, None
    try:
        header, _, b64 = url.partition(",")
        mime = header.split(";")[0].split(":")[1] if ":" in header else "image/png"
        if not b64:
            return None, None
        # Validate base64 without holding raw bytes here.
        base64.b64decode(b64[:64] + "==" if len(b64) < 64 else b64[:64])
        return mime or "image/png", b64
    except Exception:
        return None, None


def _openai_messages_to_gemini(messages):
    """Convert OpenAI chat messages to Gemini generateContent payload parts."""
    system_texts = []
    contents = []
    for m in messages or []:
        role = (m.get("role", "") or "").lower()
        text, images = _blocks_to_text_and_images(m.get("content", ""))
        if role == "system":
            if text:
                system_texts.append(text)
            continue
        parts = []
        if text:
            parts.append({"text": text})
        for mime, b64 in images:
            parts.append({"inline_data": {"mime_type": mime or "image/png",
                                          "data": b64}})
        if not parts:
            continue
        # Gemini roles: user/model. Map assistant->model, everything else->user.
        grole = "model" if role == "assistant" else "user"
        contents.append({"role": grole, "parts": parts})
    return ("\n".join(system_texts), contents)


def _call_openai(model, messages, max_tokens, temperature, json_mode,
                 timeout, retries):
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    attempt = 0
    while True:
        try:
            status, raw = _post_json(API_BASE + "/chat/completions", payload, timeout)
            break
        except ModelError as e:
            msg = str(e)
            transient = ("timeout" in msg or "connection failed" in msg
                         or msg.startswith("provider HTTP 5"))
            attempt += 1
            if transient and attempt <= retries:
                time.sleep(2)
                continue
            raise
    try:
        body = json.loads(raw.decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {}) or {}
        return content, {"prompt_tokens": int(usage.get("prompt_tokens", 0)),
                         "completion_tokens": int(usage.get("completion_tokens", 0)),
                         "total_tokens": int(usage.get("total_tokens", 0))}
    except (KeyError, IndexError, ValueError, TypeError, AttributeError):
        raise ModelError("provider returned an unreadable response") from None


def _call_gemini(model, messages, max_tokens, temperature, json_mode,
                 timeout, retries):
    system_text, contents = _openai_messages_to_gemini(messages)
    if not contents:
        raise ModelError("provider returned an unreadable response")
    # Gemini 3.x spends thinking tokens from the same maxOutputTokens budget.
    # Scale up so small max_tokens (e.g. 50-300) don't truncate JSON output.
    try:
        want = int(max_tokens or 300)
    except (TypeError, ValueError):
        want = 300
    budget = max(1024, want + 800)
    payload = {"contents": contents,
               "generationConfig": {"temperature": temperature,
                                    "maxOutputTokens": budget}}
    if system_text:
        payload["system_instruction"] = {"parts": [{"text": system_text}]}
    if json_mode:
        payload["generationConfig"]["response_mime_type"] = "application/json"
    # Localhost stub override for tests: reuse OpenAI-compatible path.
    if API_BASE.startswith("http://127.0.0.1") or API_BASE.startswith("http://localhost"):
        return _call_openai(model, messages, max_tokens, temperature,
                            json_mode, timeout, retries)
    qs = urllib.parse.urlencode({"key": _api_key()})
    url = f"{GEMINI_API_BASE}/models/{urllib.parse.quote(model, safe='')}:generateContent?{qs}"
    attempt = 0
    while True:
        try:
            status, raw = _post_json_gemini(url, payload, timeout)
            break
        except ModelError as e:
            msg = str(e)
            transient = ("timeout" in msg or "connection failed" in msg
                         or msg.startswith("provider HTTP 5"))
            attempt += 1
            if transient and attempt <= retries:
                time.sleep(2)
                continue
            raise
    try:
        body = json.loads(raw.decode("utf-8"))
        cands = body.get("candidates", []) or []
        parts = (cands[0].get("content", {}).get("parts", []) or []) if cands else []
        content = "".join([p.get("text", "") for p in parts if isinstance(p, dict)])
        if not isinstance(content, str) or not content:
            raise ValueError("empty content")
        meta = body.get("usageMetadata", {}) or {}
        pt = int(meta.get("promptTokenCount", 0) or 0)
        ct = int(meta.get("candidatesTokenCount", 0) or 0)
        tt = int(meta.get("totalTokenCount", 0) or (pt + ct))
        return content, {"prompt_tokens": pt, "completion_tokens": ct,
                         "total_tokens": tt}
    except (KeyError, IndexError, ValueError, TypeError, AttributeError):
        raise ModelError("provider returned an unreadable response") from None


def chat_complete(model, messages, max_tokens=300, temperature=0,
                  json_mode=False, timeout=60, retries=1):
    """Call chat completions. Returns (content, usage_dict).

    usage_dict has measured {prompt_tokens, completion_tokens, total_tokens}.
    Raises ModelError on any failure (auth, network, timeout, bad payload).
    """
    if not _api_key():
        raise ModelError("LLM_API_KEY is not set")
    if not model:
        raise ModelError("LLM_MODEL is not set")
    if _provider() == "gemini":
        return _call_gemini(model, messages, max_tokens, temperature,
                            json_mode, timeout, retries)
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    attempt = 0
    while True:
        try:
            status, raw = _post_json(API_BASE + "/chat/completions", payload, timeout)
            break
        except ModelError as e:
            msg = str(e)
            # Retry only transient failures, never auth errors.
            transient = ("timeout" in msg or "connection failed" in msg
                         or msg.startswith("provider HTTP 5"))
            attempt += 1
            if transient and attempt <= retries:
                time.sleep(2)
                continue
            raise
    try:
        body = json.loads(raw.decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {}) or {}
        return content, {"prompt_tokens": int(usage.get("prompt_tokens", 0)),
                         "completion_tokens": int(usage.get("completion_tokens", 0)),
                         "total_tokens": int(usage.get("total_tokens", 0))}
    except (KeyError, IndexError, ValueError, TypeError, AttributeError):
        raise ModelError("provider returned an unreadable response") from None


def image_data_url(path, max_bytes=900000):
    """Read an image file as a data URL. Returns None when unavailable."""
    try:
        raw = open(path, "rb").read()
    except OSError:
        return None
    if len(raw) > max_bytes or not raw:
        return None
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
