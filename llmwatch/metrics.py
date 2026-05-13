from typing import Dict, Any
from langchain.schema import LLMResult


# Cost per 1K tokens (USD) — updated May 2026
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    # Anthropic Claude
    "claude-opus-4-7": {"input": 0.015, "output": 0.075},
    "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5": {"input": 0.00025, "output": 0.00125},
    # Groq (hosted Llama/Mixtral)
    "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},
    "llama-3.1-8b-instant": {"input": 0.00005, "output": 0.00008},
    "llama3-70b-8192": {"input": 0.00059, "output": 0.00079},
    "llama3-8b-8192": {"input": 0.00005, "output": 0.00008},
    "mixtral-8x7b-32768": {"input": 0.00024, "output": 0.00024},
    # Google Gemini
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
}

_FALLBACK_PRICING = {"input": 0.002, "output": 0.006}


def get_model_pricing(model: str) -> Dict[str, float]:
    """Return pricing for a model, falling back to a generic rate if unknown."""
    model_lower = model.lower()
    # Sort by key length descending so more specific keys (gpt-4o-mini) match before shorter ones (gpt-4o)
    for key in sorted(MODEL_PRICING, key=len, reverse=True):
        if key in model_lower:
            return MODEL_PRICING[key]
    return _FALLBACK_PRICING


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate USD cost for a single LLM call."""
    pricing = get_model_pricing(model)
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    return round(input_cost + output_cost, 8)


def extract_token_usage(response: LLMResult) -> Dict[str, int]:
    """Pull input/output token counts from an LLMResult in a provider-agnostic way."""
    usage = {"input_tokens": 0, "output_tokens": 0}

    # LangChain stores usage in llm_output
    llm_output = response.llm_output or {}

    # OpenAI format
    token_usage = llm_output.get("token_usage", {})
    if token_usage:
        usage["input_tokens"] = token_usage.get("prompt_tokens", 0)
        usage["output_tokens"] = token_usage.get("completion_tokens", 0)
        return usage

    # Anthropic format
    if "usage" in llm_output:
        raw = llm_output["usage"]
        usage["input_tokens"] = raw.get("input_tokens", 0)
        usage["output_tokens"] = raw.get("output_tokens", 0)
        return usage

    # Fallback: try generation info on first generation
    if response.generations:
        gen_info = (response.generations[0][0].generation_info or {}) if response.generations[0] else {}
        usage["input_tokens"] = gen_info.get("prompt_tokens", 0)
        usage["output_tokens"] = gen_info.get("completion_tokens", 0)

    return usage


def cost_breakdown(model: str, input_tokens: int, output_tokens: int) -> Dict[str, Any]:
    """Return a detailed cost breakdown dict — useful for logging and dashboards."""
    pricing = get_model_pricing(model)
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    total = input_cost + output_cost
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost_usd": round(input_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "total_cost_usd": round(total, 8),
        "price_per_1k_input": pricing["input"],
        "price_per_1k_output": pricing["output"],
    }
