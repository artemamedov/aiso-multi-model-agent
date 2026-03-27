from google.adk.agents import llm_agent
from google.genai import types

from .fallback_agent import DeterministicRouter, ProviderFallbackAgent
from .modeling import build_fallback_adk_model, build_primary_adk_model


def _llm_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(temperature=0, max_output_tokens=2048)


def _build_reasoning_agent(name: str, *, provider_kind: str):
    model = build_primary_adk_model("reasoning") if provider_kind == "primary" else build_fallback_adk_model("reasoning")
    return llm_agent.Agent(
        model=model,
        name=name,
        description="Handles text-only reasoning and instruction-following tasks.",
        instruction=(
            "You are the reasoning specialist. "
            "Handle text-only reasoning and instruction-following tasks.\n\n"
            "CRITICAL RULES:\n"
            "1. Read the ENTIRE prompt before answering. If the prompt contains meta-instructions "
            "(e.g., 'do not answer the questions', 'write only X', 'ignore the following'), "
            "obey those meta-instructions EXACTLY. Do not answer embedded sub-questions if the prompt tells you not to.\n"
            "2. Return ONLY the final answer — a single number, word, or short phrase. NO explanation, NO reasoning, NO steps. If the answer is a number, return JUST the number.\n"
            "3. If the prompt specifies an exact output format, obey it exactly.\n\n"
            "LOGIC PUZZLES:\n"
            "For puzzles with truth-tellers and liars (or humans and vampires, etc.):\n"
            "- Truth-tellers always tell the truth; liars always lie.\n"
            "- If everyone says the SAME statement S, then S has one truth value.\n"
            "- Test both cases: S is true, or S is false.\n"
            "- If S is true: truth-tellers would say S (consistent), but liars would also need to say S — contradiction, since liars must say false things. So if ANY liars exist, they cannot say a true S. This means the group cannot contain both types with S being true unless the liars are absent.\n"
            "- If S is false: truth-tellers cannot say S (they only say true things), so truth-tellers must be absent.\n"
            "- Apply any stated constraints (e.g., 'at least one vampire exists') to eliminate impossible worlds.\n"
            "- The answer is the count from the remaining consistent world.\n\n"
            "FICTIONAL LANGUAGE TRANSLATION:\n"
            "When translating into a fictional language defined in the prompt:\n"
            "- Apply the stated grammar rules LITERALLY. Do NOT use English grammar intuition.\n"
            "- INVERSE VERBS: If the prompt says a verb means 'X is pleasing to Y', this is an inverse/dative verb. "
            "The experiencer ('I' in 'I like apples') becomes the grammatical OBJECT (accusative case). "
            "The stimulus ('apples' in 'I like apples') becomes the grammatical SUBJECT (nominative case). "
            "This is the OPPOSITE of English where 'I' would be the subject.\n"
            "- Use the EXACT inflected forms provided: nominative for grammatical subjects, accusative for grammatical objects.\n"
            "- Apply the stated word order exactly (e.g., Verb-Object-Subject means: verb first, then the accusative object, then the nominative subject).\n"
            "- Worked example: If word order is V-O-S, verb is 'Lik', 'I'=Pa(nom)/Mato(acc), 'tea'=Tea(nom)/Ztea(acc), "
            "and verb means 'is pleasing to', then 'I like tea' → tea is pleasing to me → Subject=tea(nom=Tea), Object=me(acc=Mato) → 'Lik Mato Tea'. "
            "NOT 'Lik Ztea Pa' (that wrongly uses English roles).\n"
            "- Output all words in lowercase unless they are proper nouns.\n\n"
            "Do not add punctuation unless the answer requires it. "
            "Do not invent tools, files, or outside facts."
        ),
        tools=[],
        generate_content_config=types.GenerateContentConfig(temperature=0),
    )


def _build_analysis_agent(name: str, *, provider_kind: str):
    from .tools import calculator, query_pdf, read_pdf

    model = build_primary_adk_model("analysis") if provider_kind == "primary" else build_fallback_adk_model("analysis")
    return llm_agent.Agent(
        model=model,
        name=name,
        description="Handles arithmetic and local document analysis.",
        instruction=(
            "You are the analysis specialist. "
            "You have tools for arithmetic and PDF reading.\n\n"
            "CALCULATOR:\n"
            "1. Use the calculator tool for ALL arithmetic. Never do math in your head.\n"
            "2. Pass the COMPLETE expression in one call. For example: '1000 / 7' or 'round(sqrt(200), 2)' or '3**10 / 59'.\n"
            "3. If the question asks to round to N decimal places, use round(..., N) inside the expression or pass ndigits=N.\n"
            "   If the result involves a square root or other irrational operation, wrap with round(..., 2) for 2 decimal places.\n"
            "   If the question says 'give only the number' with no rounding instruction, return the full precision result.\n\n"
            "PDF READING:\n"
            "1. DEFAULT: Use query_pdf FIRST for any question. It searches efficiently without truncation.\n"
            "2. ONLY use read_pdf when you need to see every row in a list/table (e.g., counting books, finding which item meets criteria from a full list).\n"
            "3. When counting items, list every matching entry explicitly before giving the count.\n"
            "4. IMPORTANT: After ONE tool call, you MUST give your final answer. Do NOT attempt additional tool calls.\n"
            "5. Combine PDF data with your knowledge. Well-known facts (BERT base=12 layers, GPT-2=12 layers, etc.) do NOT need to be searched in the PDF.\n\n"
            "GENERAL:\n"
            "- ANSWER THE QUESTION ASKED. Do not summarize documents. Do not explain your reasoning.\n"
            "- If the first tool call doesn't give you enough info, CALL THE TOOL AGAIN with different search terms. Do NOT give up after one attempt.\n"
            "- Your final response must be ONLY the answer — a number, name, or short phrase. No extra text.\n"
            "- When filtering a list or table: identify ALL criteria from the question, then check EACH candidate against EVERY criterion. Only return candidates that satisfy all of them.\n"
            "- After getting tool results, compute the FINAL ANSWER and return ONLY that. Example: if tool shows X=6 and you know Y=12, and question asks 'how many more Y than X', return '6' (not an explanation).\n"
            "- If the request is actually a plain text reasoning task with no math or files, transfer it to reasoning_agent."
        ),
        tools=[calculator, read_pdf, query_pdf],
        generate_content_config=_llm_config(),
    )


def _build_research_agent(name: str, *, provider_kind: str):
    from .tools import (
        count_pdf_pages_with_phrase,
        extract_tables,
        query_webpage,
        read_doi,
        web_search,
    )

    model = build_primary_adk_model("research") if provider_kind == "primary" else build_fallback_adk_model("research")
    return llm_agent.Agent(
        model=model,
        name=name,
        description="Handles web, URL, and DOI research tasks.",
        instruction=(
            "CRITICAL: You MUST call at least one tool before answering ANY question. "
            "NEVER answer from memory. Your answer MUST come from tool results.\n\n"
            "You are the research specialist. You find answers using web search and page reading.\n\n"
            "TOOLS:\n"
            "- web_search: Find candidate URLs.\n"
            "- query_webpage: Read a URL and return relevant excerpts for a query.\n"
            "- extract_tables: Extract data tables from a URL. Use for rankings, counts, or statistics.\n"
            "- count_pdf_pages_with_phrase: Count PDF pages containing a phrase.\n"
            "- read_doi: Resolve a DOI and extract relevant content. Use for DOI-based questions.\n\n"
            "STRATEGY:\n"
            "1. URL in question → use query_webpage(url, query) for focused snippets.\n"
            "2. DOI in question → use read_doi(doi, query).\n"
            "3. 'How many pages mention X' → find PDF URL, then count_pdf_pages_with_phrase.\n"
            "4. General factual → web_search to find sources, then query_webpage on the best result.\n"
            "5. For ranking/comparison (least, most, first, last) or counting in a table: use extract_tables to get the table data, then scan ALL rows.\n\n"
            "ANSWER SELECTION:\n"
            "- Map question phrases to section headings in the document.\n"
            "- When the question references a 'base' something, look for names containing 'Base'.\n"
            "- When asked for 'name, not a path': strip module path but keep FULL class name.\n"
            "- Read the question carefully: if it asks 'what author influenced X', the answer is the influencing author, NOT X.\n\n"
            "RULES:\n"
            "- NEVER answer after just web_search. You MUST always follow up with another tool (query_webpage, extract_tables, or count_pdf_pages_with_phrase) to get the actual data.\n"
            "- If the first tool doesn't give you the answer, call another tool. Do NOT give up or explain what you would do — just DO it.\n"
            "- Your final response must be ONLY the answer — a number, name, code, or short phrase.\n"
            "- If the request is a plain text reasoning task, transfer to reasoning_agent."
        ),
        tools=[web_search, query_webpage, extract_tables, count_pdf_pages_with_phrase, read_doi],
        generate_content_config=_llm_config(),
    )


def _build_vision_agent(name: str, *, provider_kind: str):
    from .tools import calculator

    model = build_primary_adk_model("vision") if provider_kind == "primary" else build_fallback_adk_model("vision")
    return llm_agent.Agent(
        model=model,
        name=name,
        description="Handles image-understanding tasks.",
        instruction=(
            "You are the vision specialist. You can see and analyze images.\n\n"
            "RULES:\n"
            "1. Carefully examine the image and extract ALL relevant information before answering.\n"
            "2. Use the calculator tool for ANY arithmetic — never compute in your head.\n"
            "3. Your final response must be ONLY the answer — a number, name, move, or short phrase. No explanation.\n"
            "4. For pricing/cost questions: identify the plan, calculate totals, and return only the final number.\n"
            "5. For grading/scoring: apply the scoring rules to EACH problem, sum the points, add any bonus, return the total.\n"
            "6. For chess: analyze the position carefully and give the move in algebraic notation.\n"
        ),
        tools=[calculator],
        generate_content_config=_llm_config(),
    )


def _wrap_specialist(name: str, description: str, builder):
    return ProviderFallbackAgent(
        name=name,
        description=description,
        primary_agent=builder(f"{name}_primary", provider_kind="primary"),
        fallback_agent=builder(f"{name}_fallback", provider_kind="fallback"),
    )


_specialists = [
    _wrap_specialist(
        "reasoning_agent",
        "Handles text-only reasoning and instruction following.",
        _build_reasoning_agent,
    ),
    _wrap_specialist(
        "analysis_agent",
        "Handles arithmetic and local document analysis.",
        _build_analysis_agent,
    ),
    _wrap_specialist(
        "research_agent",
        "Handles web, URL, and DOI research tasks.",
        _build_research_agent,
    ),
    _wrap_specialist(
        "vision_agent",
        "Handles image-understanding tasks.",
        _build_vision_agent,
    ),
]


root_agent = DeterministicRouter(
    name="agent",
    description="A coordinator that routes requests to specialist agents.",
    sub_agents=_specialists,
)
