from pathlib import Path
import re

import fitz


def _extract_pdf_pages(file_path: str) -> list[tuple[int, str]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file path, got: {file_path}")

    pages: list[tuple[int, str]] = []
    with fitz.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            pages.append((page_number, text))
    return pages


def read_pdf(file_path: str, max_chars: int = 16000) -> str:
    """Read a local PDF file and return full text content page by page.

    Use this tool when the user question references an attached PDF or gives you
    a filesystem path to a PDF document.

    Args:
        file_path: The local path to the PDF file.
        max_chars: Maximum number of characters to return.

    Returns:
        Extracted PDF text with page markers so you can cite or count details by page.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PDF or no text can be extracted.
    """
    page_tuples = _extract_pdf_pages(file_path)
    pages = [f"--- Page {page_number} ---\n{text}" for page_number, text in page_tuples]

    content = "\n\n".join(pages).strip()
    if not content:
        raise ValueError(f"No readable text found in PDF: {file_path}")
    if max_chars < 500:
        raise ValueError("max_chars must be at least 500.")
    if len(content) <= max_chars:
        return content
    return f"{content[:max_chars]}\n\n[TRUNCATED]"


def query_pdf(file_path: str, query: str, max_snippets: int = 10) -> str:
    """Find focused snippets in a local PDF that match a query.

    Use this after read_pdf when you need specific facts (names, statuses,
    counts, or terms) instead of the entire document text.

    Args:
        file_path: Local PDF path.
        query: What to search for inside the PDF.
        max_snippets: Maximum snippet lines to return.

    Returns:
        Page-labeled snippets that match the query terms.
    """
    if max_snippets < 1:
        raise ValueError("max_snippets must be at least 1.")

    keywords = [k for k in re.findall(r"[A-Za-z0-9']+", query.lower()) if len(k) >= 3]
    if not keywords:
        keywords = [query.strip().lower()]

    page_tuples = _extract_pdf_pages(file_path)
    matches: list[str] = []
    seen: set[str] = set()

    scored: list[tuple[int, str]] = []
    for page_number, page_text in page_tuples:
        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        for idx, line in enumerate(lines):
            lower_line = line.lower()
            score = sum(1 for kw in keywords if kw in lower_line)
            if score == 0:
                continue
            context_lines = []
            for offset in range(-1, 3):
                j = idx + offset
                if 0 <= j < len(lines):
                    context_lines.append(lines[j])
            snippet = f"Page {page_number}: {' | '.join(context_lines)}"
            if snippet not in seen:
                scored.append((score, snippet))
                seen.add(snippet)

    # Return highest-scoring snippets first
    scored.sort(key=lambda x: x[0], reverse=True)
    matches = [s for _, s in scored[:max_snippets]]

    if matches:
        return "\n".join(matches)
    return "No matching snippets found in the PDF for that query."
