import re

from google.adk.agents.readonly_context import ReadonlyContext


_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
_PDF_EXTENSIONS = (".pdf",)
_URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
_DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:a-z0-9]+\b", re.IGNORECASE)


def classify_request(user_text: str) -> str:
    lowered = user_text.lower()
    if _contains_file_extension(lowered, _IMAGE_EXTENSIONS) or any(
        token in lowered for token in ("attached image", "image below", "photo", "screenshot", "diagram")
    ):
        return "vision_agent"
    if _contains_file_extension(lowered, _PDF_EXTENSIONS) or any(
        token in lowered for token in ("attached file", "pdf", "document")
    ):
        return "analysis_agent"
    if _URL_PATTERN.search(user_text) or _DOI_PATTERN.search(user_text) or any(
        re.search(rf"\b{re.escape(token)}\b", lowered)
        for token in ("website", "web", "online", "url", "doi", "paper", "publication", "source")
    ):
        return "research_agent"
    if _needs_web_research(lowered):
        return "research_agent"
    if _needs_arithmetic(lowered):
        return "analysis_agent"
    return "reasoning_agent"


def coordinator_instruction(ctx: ReadonlyContext) -> str:
    user_text = extract_user_text(ctx)
    preferred = classify_request(user_text)
    return (
        "You are a routing coordinator for four specialist agents: "
        "reasoning_agent, analysis_agent, research_agent, and vision_agent. "
        "Never answer the user directly. "
        f"The preferred specialist for this request is {preferred}. "
        "Transfer the request immediately to the most appropriate specialist using transfer_to_agent."
    )


def extract_user_text(ctx: ReadonlyContext) -> str:
    content = getattr(ctx, "user_content", None)
    if content is None:
        return ""
    parts = getattr(content, "parts", None)
    if not parts:
        return str(content)
    text_parts: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if text:
            text_parts.append(text)
    return "\n".join(text_parts).strip()


def _needs_arithmetic(lowered: str) -> bool:
    arithmetic_keywords = (
        "divided by", "multiply", "raised to", "power of", "square root",
        "how much does each", "split it equally", "round to",
        "area in square", "calculate", "what is the sum", "what is the product",
    )
    if any(kw in lowered for kw in arithmetic_keywords):
        return True
    if re.search(r"\b\d+[\s]*[+\-*/^]\s*\d+", lowered):
        return True
    return False


def _needs_web_research(lowered: str) -> bool:
    """Detect questions that require external factual research."""
    # Questions about specific events, reports, or data that need lookup
    research_indicators = (
        "olympics", "world cup", "championship", "tournament",
        "report", "ipcc", "census", "survey",
        "according to", "published in",
        "country code", "ioc code",
    )
    if any(kw in lowered for kw in research_indicators):
        return True
    # Questions containing a specific year (1800-2099) + factual question words
    if re.search(r"\b(1[89]\d{2}|20\d{2})\b", lowered):
        fact_words = ("who", "what", "which", "how many", "least", "most", "first", "last")
        if any(fw in lowered for fw in fact_words):
            return True
    return False


def _contains_file_extension(user_text: str, suffixes: tuple[str, ...]) -> bool:
    return any(suffix in user_text for suffix in suffixes)
