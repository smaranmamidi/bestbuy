"""Token/cost accounting for model calls. Records only measured values."""
import threading


class UsageTracker:
    """Thread-safe accumulator. All token counts come from provider responses."""

    def __init__(self):
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.fallbacks = 0
        self.vision_calls = 0
        self.failures = 0

    def record_call(self, input_tokens, output_tokens, vision=False):
        with self._lock:
            self.calls += 1
            self.input_tokens += int(input_tokens or 0)
            self.output_tokens += int(output_tokens or 0)
            if vision:
                self.vision_calls += 1

    def record_fallback(self):
        with self._lock:
            self.fallbacks += 1

    def record_failure(self):
        with self._lock:
            self.failures += 1

    def snapshot(self):
        with self._lock:
            total = self.input_tokens + self.output_tokens
            return {"calls": self.calls,
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                    "total_tokens": total,
                    "fallbacks": self.fallbacks,
                    "vision_calls": self.vision_calls,
                    "failures": self.failures}

    def write_report(self, path, model_desc, num_requests, extra_lines=None):
        """Write evaluation/usage_report.md. Contains no secrets by construction."""
        snap = self.snapshot()
        avg = snap["total_tokens"] / num_requests if num_requests else 0
        lines = [
            "# Token usage report",
            "",
            "Final full dataset run that generated output.csv through the AI-agent pipeline.",
            "",
            "## Summary",
            "",
            f"- Provider: {model_desc.get('provider', 'none')}",
            f"- Model: {model_desc.get('model', 'none')}",
            f"- Number of model calls: {snap['calls']}",
            f"- Input tokens: {snap['input_tokens']}",
            f"- Output tokens: {snap['output_tokens']}",
            f"- Total tokens: {snap['total_tokens']}",
            f"- Average tokens per request ({num_requests} requests): {avg:.1f}",
            "- Estimated total cost: see provider billing for the configured model",
            "- Estimated cost per request: see provider billing for the configured model",
            f"- Requests served by deterministic fallback: {snap['fallbacks']}",
            f"- Vision calls: {snap['vision_calls']}",
            f"- Failed model calls: {snap['failures']}",
            "",
            "## Per model statistics",
            "",
            f"- {model_desc.get('model', 'none')}: {snap['calls']} calls, "
            f"{snap['input_tokens']} input tokens, {snap['output_tokens']} output tokens",
            "",
            "Token counts are measured values returned by the provider with each",
            "response, summed across the run. They are never estimated or fabricated.",
            "Fallback rows reuse the deterministic engine and add zero tokens.",
            "",
            "## Reproduce",
            "",
            "Set `LLM_API_KEY` and `LLM_MODEL` in the environment (or a local `.env`",
            "file, see `.env.example`), then run `python3 code/main.py` from the",
            "repository root. Without a key the pipeline uses the deterministic",
            "fallback for every request.",
            "",
        ]
        if extra_lines:
            lines.extend(extra_lines)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
