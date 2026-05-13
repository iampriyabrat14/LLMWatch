"""
Inline per-query RAGAS-style evaluation using Groq synchronously.
Implements Faithfulness + Context Relevance as LLM-as-judge — same
concept as RAGAS but fully synchronous, no event loop issues.
Uses llama-3.1-8b-instant for eval to avoid rate-limit conflicts
with the main 70B model used for answer generation.
"""
import os
import re
import time
from typing import List
from groq import Groq

_client: Groq = None

_EVAL_MODEL = "llama-3.1-8b-instant"


def _get_client() -> Groq:
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            # fallback: try loading .env relative to this file
            from dotenv import load_dotenv
            load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"), override=True)
            key = os.getenv("GROQ_API_KEY")
        _client = Groq(api_key=key)
    return _client


def _ask(prompt: str) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=_EVAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=64,
    )
    return resp.choices[0].message.content.strip()


def _parse_score(text: str) -> float:
    """Extract first float/int in 0-1 range from LLM response."""
    matches = re.findall(r"0?\.\d+|1\.0|[01]", text)
    for m in matches:
        try:
            v = float(m)
            if 0.0 <= v <= 1.0:
                return round(v, 4)
        except ValueError:
            continue
    return 0.5  # neutral fallback


def _faithfulness_score(question: str, answer: str, contexts: List[str]) -> float:
    context_str = "\n".join(contexts)
    prompt = f"""You are evaluating if an answer is faithful to the given context.

CONTEXT:
{context_str}

QUESTION: {question}
ANSWER: {answer}

Score how faithful the answer is to the context ONLY (0.0 = completely hallucinated, 1.0 = fully grounded in context).
Respond with ONLY a number between 0.0 and 1.0."""
    return _parse_score(_ask(prompt))


def _context_relevance_score(question: str, contexts: List[str]) -> float:
    context_str = "\n".join(contexts)
    prompt = f"""You are evaluating if retrieved context is relevant to a question.

QUESTION: {question}

RETRIEVED CONTEXT:
{context_str}

Score how relevant the context is for answering the question (0.0 = completely irrelevant, 1.0 = perfectly relevant).
Respond with ONLY a number between 0.0 and 1.0."""
    return _parse_score(_ask(prompt))


def evaluate_query(
    question: str,
    answer: str,
    contexts: List[str],
    **_kwargs,
) -> dict:
    """
    Run faithfulness + context relevance evaluation on a single query.
    Returns dict with individual scores and overall average.
    """
    try:
        faithfulness      = _faithfulness_score(question, answer, contexts)
        time.sleep(1)  # avoid Groq rate limits between eval calls
        context_relevance = _context_relevance_score(question, contexts)
        overall           = round((faithfulness + context_relevance) / 2, 4)

        return {
            "faithfulness":      faithfulness,
            "context_relevance": context_relevance,
            "overall":           overall,
        }

    except Exception as e:
        return {
            "faithfulness":      0.0,
            "context_relevance": 0.0,
            "overall":           0.0,
            "error":             str(e),
        }
