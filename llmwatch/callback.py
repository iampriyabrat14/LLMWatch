import time
import uuid
from typing import Any, Dict, List, Optional, Union
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

from .metrics import calculate_cost, extract_token_usage
from .db import MetricsDB


class LLMWatchCallback(BaseCallbackHandler):
    """Drop-in LangChain callback handler that tracks latency, cost, and token usage."""

    def __init__(
        self,
        project: str = "default",
        db_path: str = "llmwatch.db",
        budget_limit_usd: Optional[float] = None,
        latency_alert_ms: Optional[float] = None,
    ):
        self.project = project
        self.db = MetricsDB(db_path)
        self.budget_limit_usd = budget_limit_usd
        self.latency_alert_ms = latency_alert_ms
        self._run_start_times: Dict[str, float] = {}
        self._run_models: Dict[str, str] = {}

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        self._run_start_times[str(run_id)] = time.perf_counter()
        model = (
            serialized.get("kwargs", {}).get("model_name")
            or serialized.get("kwargs", {}).get("model")
            or "unknown"
        )
        self._run_models[str(run_id)] = model

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        run_key = str(run_id)
        latency_ms = self._finish_timer(run_key)
        model = self._run_models.pop(run_key, "unknown")

        token_usage = extract_token_usage(response)
        input_tokens = token_usage["input_tokens"]
        output_tokens = token_usage["output_tokens"]
        cost_usd = calculate_cost(model, input_tokens, output_tokens)

        self.db.insert_metric(
            project=self.project,
            model=model,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            error=None,
        )

        self._check_alerts(latency_ms, cost_usd)

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        run_key = str(run_id)
        latency_ms = self._finish_timer(run_key)
        model = self._run_models.pop(run_key, "unknown")

        self.db.insert_metric(
            project=self.project,
            model=model,
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            error=str(error),
        )

    def _finish_timer(self, run_key: str) -> float:
        start = self._run_start_times.pop(run_key, None)
        if start is None:
            return 0.0
        return (time.perf_counter() - start) * 1000  # ms

    def _check_alerts(self, latency_ms: float, cost_usd: float) -> None:
        if self.latency_alert_ms and latency_ms > self.latency_alert_ms:
            print(
                f"[LLMWatch] LATENCY ALERT: {latency_ms:.1f}ms exceeds "
                f"threshold {self.latency_alert_ms}ms"
            )
        if self.budget_limit_usd:
            cumulative = self.db.get_cumulative_cost(self.project)
            if cumulative > self.budget_limit_usd:
                print(
                    f"[LLMWatch] BUDGET ALERT: ${cumulative:.4f} exceeds "
                    f"limit ${self.budget_limit_usd:.2f}"
                )

    def summary(self) -> Dict[str, Any]:
        """Return a quick summary dict of metrics for the current project."""
        return self.db.get_summary(self.project)
