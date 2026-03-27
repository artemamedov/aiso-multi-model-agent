import os
from typing import Any


DEFAULT_GOOGLE_TEXT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_GOOGLE_VISION_MODEL = "gemini-2.5-flash-lite"
DEFAULT_OLLAMA_TEXT_MODEL = "qwen3:14b"
DEFAULT_OLLAMA_VISION_MODEL = "qwen2.5vl:7b"


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return str(value).strip().strip('"').strip("'")


def primary_provider() -> str:
    provider = _env("PRIMARY_MODEL_PROVIDER") or _env("MODEL_PROVIDER", "ollama")
    provider = provider.lower()
    if provider not in {"ollama", "gemini"}:
        raise ValueError(f"Unsupported primary provider: {provider}")
    return provider


def fallback_provider() -> str:
    provider = _env("FALLBACK_MODEL_PROVIDER")
    if not provider:
        provider = "gemini" if primary_provider() == "ollama" else "ollama"
    provider = provider.lower()
    if provider not in {"ollama", "gemini"}:
        raise ValueError(f"Unsupported fallback provider: {provider}")
    return provider


def ollama_api_base() -> str:
    return _env("OLLAMA_API_BASE", "http://127.0.0.1:11434").rstrip("/")


def _role_key(role: str) -> str:
    return role.upper().replace("-", "_")


def _provider_override(kind: str, role: str) -> str:
    role_value = _env(f"{kind}_{_role_key(role)}_PROVIDER")
    if role_value:
        role_value = role_value.lower()
        if role_value not in {"ollama", "gemini"}:
            raise ValueError(f"Unsupported {kind.lower()} provider override for role {role}: {role_value}")
    return role_value


def _model_override(kind: str, role: str) -> str:
    return _env(f"{kind}_{_role_key(role)}_MODEL")


def _primary_text_model() -> str:
    return (
        _env("PRIMARY_TEXT_MODEL")
        or _env("OLLAMA_FAST_MODEL")
        or _env("OLLAMA_COMPLEX_MODEL")
        or DEFAULT_OLLAMA_TEXT_MODEL
    )


def _primary_vision_model() -> str:
    return _env("PRIMARY_VISION_MODEL") or _env("OLLAMA_VISION_MODEL") or DEFAULT_OLLAMA_VISION_MODEL


def _fallback_text_model() -> str:
    return (
        _env("FALLBACK_TEXT_MODEL")
        or _env("GOOGLE_FAST_MODEL")
        or _env("GOOGLE_COMPLEX_MODEL")
        or DEFAULT_GOOGLE_TEXT_MODEL
    )


def _fallback_vision_model() -> str:
    return _env("FALLBACK_VISION_MODEL") or _env("GOOGLE_VISION_MODEL") or DEFAULT_GOOGLE_VISION_MODEL


def _build_adk_model(*, provider: str, model_name: str, role: str, think: bool = True):
    if provider == "gemini":
        return model_name

    from google.adk.models.lite_llm import LiteLlm

    api_base = ollama_api_base()
    model_prefix = "ollama_chat"

    # LM Studio support: use OpenAI-compatible API
    use_lmstudio = _env("LLM_BACKEND") == "lmstudio"
    if use_lmstudio:
        api_base = _env("LMSTUDIO_API_BASE", "http://127.0.0.1:1234/v1")
        model_prefix = "openai"

    additional_args: dict[str, Any] = {
        "api_base": api_base,
        "think": think if not use_lmstudio else False,
    }
    if use_lmstudio:
        additional_args["api_key"] = "lm-studio"
        additional_args["drop_params"] = True
    timeout_seconds = _env("OLLAMA_TIMEOUT_SECONDS")
    if timeout_seconds:
        try:
            additional_args["timeout"] = int(timeout_seconds)
        except ValueError:
            pass
    return LiteLlm(model=f"{model_prefix}/{model_name}", **additional_args)


def build_primary_adk_model(role: str = "text", *, think: bool = True):
    provider = _provider_override("PRIMARY", role) or primary_provider()
    if provider == "gemini":
        model_name = DEFAULT_GOOGLE_VISION_MODEL if role == "vision" else DEFAULT_GOOGLE_TEXT_MODEL
        model_name = _model_override("PRIMARY", role) or (
            _env("PRIMARY_VISION_MODEL", model_name) if role == "vision" else _env("PRIMARY_TEXT_MODEL", model_name)
        )
        return _build_adk_model(provider="gemini", model_name=model_name, role=role, think=think)

    model_name = _model_override("PRIMARY", role) or (_primary_vision_model() if role == "vision" else _primary_text_model())
    return _build_adk_model(provider="ollama", model_name=model_name, role=role, think=think)


def build_fallback_adk_model(role: str = "text", *, think: bool = True):
    provider = _provider_override("FALLBACK", role) or fallback_provider()
    if provider == "ollama":
        model_name = _model_override("FALLBACK", role) or (_primary_vision_model() if role == "vision" else _primary_text_model())
        return _build_adk_model(provider="ollama", model_name=model_name, role=role, think=think)

    model_name = _model_override("FALLBACK", role) or (_fallback_vision_model() if role == "vision" else _fallback_text_model())
    return _build_adk_model(provider="gemini", model_name=model_name, role=role, think=think)


def should_fallback_on_error(error: Exception) -> bool:
    message = str(error).lower()
    tokens = (
        "429",
        "resource_exhausted",
        "rate limit",
        "quota",
        "api_key_invalid",
        "invalid_argument",
        "connection aborted",
        "connection error",
        "timed out",
        "timeout",
        "remote end closed connection",
        "internal server error",
        "service unavailable",
    )
    return any(token in message for token in tokens)
