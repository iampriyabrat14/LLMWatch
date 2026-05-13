import uuid
import pytest
from unittest.mock import MagicMock, patch
from langchain.schema import LLMResult, Generation

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llmwatch.callback import LLMWatchCallback
from llmwatch.metrics import calculate_cost, extract_token_usage, get_model_pricing
from llmwatch.db import MetricsDB


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def callback(tmp_db):
    return LLMWatchCallback(project="test", db_path=tmp_db)


def make_llm_result(prompt_tokens=100, completion_tokens=50):
    return LLMResult(
        generations=[[Generation(text="Test response")]],
        llm_output={"token_usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}},
    )


# ── Cost calculation tests ─────────────────────────────────────────────────────

def test_calculate_cost_gpt4o():
    cost = calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
    expected = (1000 / 1000 * 0.005) + (500 / 1000 * 0.015)
    assert abs(cost - expected) < 1e-7


def test_calculate_cost_zero_tokens():
    assert calculate_cost("gpt-4o", 0, 0) == 0.0


def test_calculate_cost_unknown_model():
    cost = calculate_cost("unknown-model-xyz", 1000, 500)
    assert cost > 0  # uses fallback pricing


def test_get_model_pricing_partial_match():
    pricing = get_model_pricing("openai/gpt-4o-mini")
    assert pricing["input"] == 0.00015


# ── Token extraction tests ─────────────────────────────────────────────────────

def test_extract_token_usage_openai_format():
    result = make_llm_result(100, 50)
    usage = extract_token_usage(result)
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50


def test_extract_token_usage_anthropic_format():
    result = LLMResult(
        generations=[[Generation(text="hi")]],
        llm_output={"usage": {"input_tokens": 80, "output_tokens": 30}},
    )
    usage = extract_token_usage(result)
    assert usage["input_tokens"] == 80
    assert usage["output_tokens"] == 30


def test_extract_token_usage_empty():
    result = LLMResult(generations=[[Generation(text="hi")]], llm_output={})
    usage = extract_token_usage(result)
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0


# ── DB tests ──────────────────────────────────────────────────────────────────

def test_db_insert_and_retrieve(tmp_db):
    db = MetricsDB(tmp_db)
    db.insert_metric("proj", "gpt-4o", 250.0, 100, 50, 0.00125)
    rows = db.get_metrics("proj")
    assert len(rows) == 1
    assert rows[0]["latency_ms"] == 250.0
    assert rows[0]["input_tokens"] == 100


def test_db_percentiles(tmp_db):
    db = MetricsDB(tmp_db)
    latencies = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for lat in latencies:
        db.insert_metric("proj", "gpt-4o", lat, 50, 25, 0.001)
    pcts = db.get_percentiles("proj")
    assert pcts["p50_ms"] > 0
    assert pcts["p95_ms"] >= pcts["p50_ms"]


def test_db_cumulative_cost(tmp_db):
    db = MetricsDB(tmp_db)
    db.insert_metric("proj", "gpt-4o", 200.0, 100, 50, 0.01)
    db.insert_metric("proj", "gpt-4o", 300.0, 100, 50, 0.02)
    total = db.get_cumulative_cost("proj")
    assert abs(total - 0.03) < 1e-6


def test_db_summary_empty(tmp_db):
    db = MetricsDB(tmp_db)
    summary = db.get_summary("nonexistent")
    assert summary["total_queries"] == 0
    assert summary["total_cost_usd"] == 0.0


# ── Callback integration tests ────────────────────────────────────────────────

def test_callback_records_metric_on_llm_end(callback):
    run_id = uuid.uuid4()
    serialized = {"kwargs": {"model_name": "gpt-4o-mini"}}
    callback.on_llm_start(serialized, ["Hello"], run_id=run_id)
    callback.on_llm_end(make_llm_result(100, 50), run_id=run_id)

    rows = callback.db.get_metrics("test")
    assert len(rows) == 1
    assert rows[0]["model"] == "gpt-4o-mini"
    assert rows[0]["latency_ms"] > 0
    assert rows[0]["input_tokens"] == 100
    assert rows[0]["output_tokens"] == 50
    assert rows[0]["cost_usd"] > 0


def test_callback_records_error(callback):
    run_id = uuid.uuid4()
    callback.on_llm_start({"kwargs": {}}, ["Hello"], run_id=run_id)
    callback.on_llm_error(RuntimeError("API timeout"), run_id=run_id)

    rows = callback.db.get_metrics("test")
    assert rows[0]["error"] == "API timeout"
    assert rows[0]["cost_usd"] == 0.0


def test_callback_budget_alert_prints(tmp_db, capsys):
    cb = LLMWatchCallback(project="budget-test", db_path=tmp_db, budget_limit_usd=0.001)
    run_id = uuid.uuid4()
    cb.on_llm_start({"kwargs": {"model_name": "gpt-4o"}}, ["q"], run_id=run_id)
    cb.on_llm_end(make_llm_result(10000, 5000), run_id=run_id)
    captured = capsys.readouterr()
    assert "BUDGET ALERT" in captured.out


def test_callback_latency_alert_prints(tmp_db, capsys):
    cb = LLMWatchCallback(project="lat-test", db_path=tmp_db, latency_alert_ms=0.001)
    run_id = uuid.uuid4()
    cb.on_llm_start({"kwargs": {}}, ["q"], run_id=run_id)
    cb.on_llm_end(make_llm_result(10, 5), run_id=run_id)
    captured = capsys.readouterr()
    assert "LATENCY ALERT" in captured.out
