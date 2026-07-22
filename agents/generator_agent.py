"""
generator_agent.py
------------------
LangGraph node: Generator Agent

Responsibilities:
  1. Construct a prompt from the system template (config.yaml) and the
     retrieved context.
  2. Call the configured LLM (OpenAI, Azure OpenAI, or Anthropic).
  3. Instruct the model to include inline citations in every response.
  4. Support retry — on subsequent attempts the temperature is nudged upward
     to encourage a different formulation.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


# ── LLM client factory ────────────────────────────────────────────────────────

def _get_openai_client(api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key)


def _get_azure_client(api_key: str):
    from openai import AzureOpenAI
    config = get_config()
    endpoint = os.environ.get(config.llm.azure_endpoint_env, "")
    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=config.llm.azure_api_version,
    )


def _get_anthropic_client(api_key: str):
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def _build_llm_client():
    config = get_config()
    provider = config.llm.provider
    api_key_env = config.llm.api_key_env

    if provider == "anthropic":
        api_key_env = config.llm.get("anthropic_api_key_env", "ANTHROPIC_API_KEY")

    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise EnvironmentError(
            f"LLM API key not set. Please set the '{api_key_env}' environment variable."
        )

    if provider == "openai":
        return _get_openai_client(api_key), "openai"
    elif provider == "azure":
        return _get_azure_client(api_key), "azure"
    elif provider == "anthropic":
        return _get_anthropic_client(api_key), "anthropic"
    else:
        raise NotImplementedError(f"LLM provider '{provider}' is not supported.")


# ── Generator node ────────────────────────────────────────────────────────────

def generator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generator Agent node.

    Reads  : state["query"], state["context"], state["retry_count"]
    Writes : state["draft_response"], state["retry_count"]
    """
    config = get_config()
    query: str = state["query"]
    context: str = state.get("context", "")
    retry_count: int = state.get("retry_count", 0)
    chat_history: list = state.get("chat_history", [])

    logger.info({"event": "generator_start", "attempt": retry_count + 1, "history_turns": len(chat_history)})

    # If retriever found nothing, return the off-topic fallback immediately.
    if not context.strip():
        fallback = config.guardrails.off_topic_response
        return {
            "draft_response": fallback,
            "final_response": fallback,
            "retry_count": retry_count + 1,
        }

    try:
        client, provider = _build_llm_client()
        system_prompt: str = config.prompts.system

        user_prompt = (
            f"Context:\n{context}\n\n"
            "─────────────────────────────────────────────\n"
            f"Question: {query}\n\n"
            "Instructions:\n"
            "• Base your answer ONLY on the context provided above.\n"
            "• After each factual statement, add an inline citation:\n"
            "  [Source: <filename>, Page <number>]\n"
            "• If the context is insufficient, say so explicitly.\n"
            "• Do not speculate or add information from outside the context."
        )

        # Slightly raise temperature on retries to encourage reformulation
        temperature = min(config.llm.temperature + retry_count * 0.1, 0.8)

        # ── OpenAI / Azure ────────────────────────────────────────────────────
        if provider in ("openai", "azure"):
            model = (
                config.llm.azure_deployment
                if provider == "azure"
                else config.llm.model
            )
            # Build messages: system → history (last 6 msgs = 3 turns) → current
            messages = [{"role": "system", "content": system_prompt}]
            for turn in chat_history[-6:]:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": user_prompt})

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=config.llm.max_tokens,
            )
            draft = response.choices[0].message.content

        # ── Anthropic ─────────────────────────────────────────────────────────
        elif provider == "anthropic":
            # Build history for Anthropic format
            anthropic_messages = []
            for turn in chat_history[-6:]:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    anthropic_messages.append({"role": role, "content": content})
            anthropic_messages.append({"role": "user", "content": user_prompt})

            response = client.messages.create(
                model=config.llm.model,
                max_tokens=config.llm.max_tokens,
                system=system_prompt,
                messages=anthropic_messages,
            )
            draft = response.content[0].text

        else:
            raise NotImplementedError(f"Provider '{provider}' not handled in generator.")

        logger.info({"event": "generator_complete", "draft_length": len(draft)})
        return {"draft_response": draft, "retry_count": retry_count + 1}

    except Exception as exc:
        logger.error({"event": "generator_error", "error": str(exc)}, exc_info=True)
        return {
            "draft_response": "An error occurred while generating the response.",
            "retry_count": retry_count + 1,
            "error": str(exc),
        }
