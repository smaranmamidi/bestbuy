"""Secure LLM configuration. Environment variables only, never hardcoded keys.

Reads (in order of precedence): process environment, then a local `.env`
file in the repository root. Values are never printed, logged, or included
in errors. Use `is_configured()` to check availability without touching
the secret itself.
"""
import os
from pathlib import Path

ENV_FILE_NAMES = (".env",)
RECOGNIZED_KEYS = (
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "LLM_MODEL",
    "VISION_ENABLED",
    "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_TOKENS",
)


def load_dotenv(repo_root):
    """Minimal `.env` parser (stdlib only). Never overrides real env vars."""
    for name in ENV_FILE_NAMES:
        path = Path(repo_root) / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key not in RECOGNIZED_KEYS:
                continue
            value = value.strip().strip("'").strip('"')
            os.environ.setdefault(key, value)


class LLMConfig:
    """Resolved, redaction-safe LLM configuration."""

    def __init__(self, repo_root=None):
        if repo_root is not None:
            load_dotenv(repo_root)
        self.provider = (os.environ.get("LLM_PROVIDER", "") or "").strip().lower()
        self.model = (os.environ.get("LLM_MODEL", "") or "").strip()
        self.vision_enabled = (os.environ.get("VISION_ENABLED", "true") or "").strip().lower() not in (
            "0", "false", "no", "off", "")
        try:
            self.timeout = int(os.environ.get("LLM_TIMEOUT_SECONDS", "60") or 60)
        except ValueError:
            self.timeout = 60
        try:
            self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "300") or 300)
        except ValueError:
            self.max_tokens = 300

    def is_configured(self):
        """True only when a usable key and model are present. Never reveals them."""
        return (self.provider in ("openai", "gemini")
                and bool((os.environ.get("LLM_API_KEY", "") or "").strip())
                and bool(self.model))

    def missing_reason(self):
        """Safe, secret-free explanation of why the model path is unavailable."""
        if self.provider not in ("openai", "gemini"):
            return "LLM_PROVIDER is not 'openai' or 'gemini'"
        if not (os.environ.get("LLM_API_KEY", "") or "").strip():
            return "LLM_API_KEY is not set"
        if not self.model:
            return "LLM_MODEL is not set"
        return ""

    def describe(self):
        """Public, secret-free summary for logs and reports."""
        return {"provider": self.provider or "none",
                "model": self.model or "none",
                "vision_enabled": self.vision_enabled,
                "configured": self.is_configured()}
