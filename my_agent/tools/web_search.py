from io import BytesIO
import re
from urllib.parse import urlparse

import fitz
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
)


def _query_keywords(query: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9']+", query.lower())
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "what",
        "which",
        "when",
        "where",
        "have",
        "has",
        "had",
        "are",
        "was",
        "were",
        "only",
        "give",
        "just",
        "name",
    }
    keywords = [w for w in words if len(w) >= 3 and w not in stop]
    return keywords[:12]


def _extract_relevant_excerpt(content: str, query: str, max_chars: int = 900) -> str:
    keywords = _query_keywords(query)
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return content[:max_chars]

    scored: list[tuple[int, int]] = []
    for idx, line in enumerate(lines):
        lower = line.lower()
        score = sum(1 for keyword in keywords if keyword in lower)
        if score > 0:
            scored.append((score, idx))

    if not scored:
        return "\n".join(lines)[:max_chars]

    scored.sort(reverse=True)
    best_idx = scored[0][1]
    start = max(0, best_idx - 2)
    end = min(len(lines), best_idx + 3)
    excerpt = "\n".join(lines[start:end])
    return excerpt[:max_chars]


def _tables_to_markdown(soup):
    """Convert HTML tables to pipe-delimited text for readability."""
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [
                cell.get_text(strip=True).replace("|", "/")
                for cell in tr.find_all(["td", "th"])
            ]
            if cells:
                rows.append(" | ".join(cells))
        if rows:
            wrapper = soup.new_tag("div")
            wrapper.string = "\n".join(rows)
            table.replace_with(wrapper)
        else:
            table.decompose()


def _extract_section(soup, fragment):
    """Extract section content from a fragment ID through the next same-level heading."""
    target = soup.find(id=fragment)
    if target is None:
        return ""
    heading = target
    if heading.name not in ("h1", "h2", "h3", "h4", "h5", "h6"):
        heading = target.find_parent(["h1", "h2", "h3", "h4", "h5", "h6"])
    if not heading:
        return target.get_text(separator="\n", strip=True)
    level = int(heading.name[1])
    parts = [heading.get_text(strip=True)]
    for sibling in heading.find_next_siblings():
        if sibling.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            if int(sibling.name[1]) <= level:
                break
        text = sibling.get_text(separator="\n", strip=True)
        if text:
            parts.append(text)
    return "\n".join(parts)


def web_search(query: str, max_results: int = 8) -> str:
    """Search the web and return ranked result metadata.

    Use this when you first need candidate URLs.
    """
    if max_results < 1:
        raise ValueError("max_results must be at least 1.")
    max_results = min(max_results, 5)

    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)

    if not results:
        return "No search results found."

    formatted_results: list[str] = []
    for index, result in enumerate(results, start=1):
        title = result.get("title", "").strip()
        url = result.get("href", "").strip()
        snippet = result.get("body", "").strip()[:220]
        formatted_results.append(
            f"{index}. Title: {title}\nURL: {url}\nSnippet: {snippet}"
        )
    return "\n\n".join(formatted_results)


def fetch_webpage(url: str, max_chars: int = 12000) -> str:
    """Fetch and read a webpage or PDF URL.

    Use this when a question includes a URL or after selecting a search result.
    """
    if max_chars < 500:
        raise ValueError("max_chars must be at least 500.")

    parsed_url = urlparse(url)
    request_url = parsed_url._replace(fragment="").geturl()

    try:
        response = requests.get(
            request_url,
            timeout=20,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
    except requests.RequestException as error:
        return f"Fetch error: {error}"

    content_type = response.headers.get("content-type", "").lower()
    is_pdf = "pdf" in content_type or parsed_url.path.lower().endswith(".pdf")

    if is_pdf:
        if not response.content:
            return "The URL looked like a PDF, but it returned an empty file."
        pages: list[str] = []
        try:
            with fitz.open(stream=BytesIO(response.content), filetype="pdf") as document:
                for page_number, page in enumerate(document, start=1):
                    text = page.get_text("text").strip()
                    if text:
                        pages.append(f"--- Page {page_number} ---\n{text}")
        except Exception:
            return "The URL looked like a PDF, but readable PDF text could not be extracted."
        if not pages:
            return "The PDF was fetched successfully, but no readable text was found."
        return "\n\n".join(pages)[:max_chars]

    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        element.decompose()
    for element in soup.find_all(attrs={"role": "navigation"}):
        element.decompose()
    for element in soup.find_all(class_=re.compile(r"navbox|catlinks|sidebar|noprint|mw-authority")):
        element.decompose()
    _tables_to_markdown(soup)

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    section_text = ""
    if parsed_url.fragment:
        section_text = _extract_section(soup, parsed_url.fragment)

    main = soup.find("main")
    root = main if main is not None else soup.body if soup.body is not None else soup
    text = section_text if section_text else root.get_text(separator="\n")
    cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned_text = "\n".join(cleaned_lines)

    if title:
        cleaned_text = f"Title: {title}\n\n{cleaned_text}"
    return cleaned_text[:max_chars]


def query_webpage(url: str, query: str, max_snippets: int = 8) -> str:
    """Read a specific web source and return the most relevant excerpts.

    Use this after selecting a URL when you need focused evidence for a query
    rather than the full page text.
    """
    if max_snippets < 1:
        raise ValueError("max_snippets must be at least 1.")

    lowered_url = url.lower()
    max_chars = 250000 if lowered_url.endswith(".pdf") or ".pdf" in lowered_url else 120000
    content = fetch_webpage(url, max_chars=max_chars)
    if content.startswith("Fetch error:"):
        return content

    keywords = _query_keywords(query)
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return "No readable text found."

    window_size = 5
    scored: list[tuple[int, int]] = []
    for idx in range(len(lines)):
        window = " ".join(lines[idx:idx + window_size]).lower()
        score = sum(1 for keyword in keywords if keyword in window)
        if score > 0:
            scored.append((score, idx))

    if not scored:
        return content[:1200]

    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    snippets: list[str] = []
    used_ranges: list[tuple[int, int]] = []
    for _, idx in scored:
        start = max(0, idx - 1)
        end = min(len(lines), idx + window_size + 1)
        # If snippet touches a table (pipe-delimited rows), include the full table
        if any(" | " in lines[i] for i in range(start, min(end, len(lines)))):
            while start > 0 and " | " in lines[start - 1]:
                start -= 1
            while end < len(lines) and " | " in lines[end]:
                end += 1
            end = min(end, start + 120)
        if any(start < ur_end and end > ur_start for ur_start, ur_end in used_ranges):
            continue
        used_ranges.append((start, end))
        snippets.append("\n".join(lines[start:end]))
        if len(snippets) >= max_snippets:
            break
    return "\n\n".join(snippets)


def web_research(
    query: str,
    max_results: int = 5,
    max_chars_per_page: int = 3500,
) -> str:
    """Search and read top pages, returning focused evidence snippets.

    Use this when flash-lite needs compact evidence instead of full page dumps.
    """
    if max_results < 1:
        raise ValueError("max_results must be at least 1.")
    max_results = min(max_results, 4)

    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
    if not results:
        return "No research results found."

    sections: list[str] = []
    for index, result in enumerate(results, start=1):
        title = result.get("title", "").strip()
        url = result.get("href", "").strip()
        snippet = result.get("body", "").strip()

        section_lines = [
            f"Result {index}",
            f"Title: {title}",
            f"URL: {url}",
            f"Search Snippet: {snippet}",
        ]
        if url:
            try:
                evidence = query_webpage(url, query, max_snippets=3)
                section_lines.append(f"Evidence Excerpt:\n{evidence}")
            except Exception as error:
                section_lines.append(f"Fetch Error: {error}")
        sections.append("\n".join(section_lines))

    return "\n\n".join(sections)


def extract_tables(url: str, query: str = "", max_chars: int = 8000) -> str:
    """Extract data tables from a webpage in a compact format.

    Use this when you need to read, compare, or analyze tabular data such as
    rankings, statistics, counts, or lists from a web page. Pass a query to
    rank tables by relevance.
    """
    if max_chars < 500:
        raise ValueError("max_chars must be at least 500.")

    parsed_url = urlparse(url)
    request_url = parsed_url._replace(fragment="").geturl()

    try:
        response = requests.get(request_url, timeout=20, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
    except requests.RequestException as error:
        return f"Fetch error: {error}"

    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        element.decompose()
    for element in soup.find_all(attrs={"role": "navigation"}):
        element.decompose()
    for element in soup.find_all(class_=re.compile(r"navbox|catlinks|sidebar|noprint|mw-authority")):
        element.decompose()

    tables: list[str] = []
    for table in soup.find_all("table"):
        rows: list[str] = []
        for tr in table.find_all("tr"):
            cells = [cell.get_text(strip=True).replace("|", "/") for cell in tr.find_all(["td", "th"])]
            if cells:
                rows.append(" | ".join(cells))
        if len(rows) >= 2:
            tables.append("\n".join(rows))

    if not tables:
        return "No data tables found on this page."

    if query:
        keywords = _query_keywords(query)
        scored = []
        for table_text in tables:
            lower = table_text.lower()
            score = sum(1 for kw in keywords if kw in lower)
            scored.append((score, table_text))
        scored.sort(key=lambda x: x[0], reverse=True)
        tables = [t for _, t in scored]

    return "\n\n---\n\n".join(tables)[:max_chars]


def count_pdf_pages_with_phrase(url: str, phrase: str) -> str:
    """Count how many pages in a PDF URL contain a phrase.

    Use this for questions like "how many pages mention X".
    """
    if not phrase.strip():
        raise ValueError("phrase must not be empty.")

    try:
        response = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
    except requests.RequestException as error:
        return f"Fetch error: {error}"

    content_type = response.headers.get("content-type", "").lower()
    parsed_url = urlparse(url)
    is_pdf = "pdf" in content_type or parsed_url.path.lower().endswith(".pdf")
    if not is_pdf:
        return "Error: The provided URL does not appear to be a PDF."

    target = phrase.lower()
    matching_pages: list[int] = []
    with fitz.open(stream=BytesIO(response.content), filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").lower()
            if target in text:
                matching_pages.append(page_number)

    if not matching_pages:
        return "0"
    return str(len(matching_pages))


def read_doi(doi: str, query: str) -> str:
    """Resolve a DOI and return relevant excerpts from the resource.

    Use this when the question references a DOI. It resolves the DOI, finds
    an open-access version, and extracts content matching the query.
    """
    if not doi.strip():
        raise ValueError("doi must not be empty.")
    if not query.strip():
        raise ValueError("query must not be empty.")

    doi = doi.strip()

    # Step 1: resolve DOI to get metadata and co-access links
    doi_url = f"https://doi.org/{doi}"
    metadata_text = fetch_webpage(doi_url, max_chars=8000)

    # Extract candidate URLs from the DOI metadata page
    candidate_urls: list[str] = []
    title = ""
    for line in metadata_text.splitlines():
        stripped = line.strip()
        # Extract title (first non-URL, non-metadata line)
        if not title and stripped and not stripped.startswith(("http", "{", "debug", "Record", "The publisher", "DOI", "Published")):
            title = stripped
        # Collect URLs found in the metadata
        for match in re.finditer(r"https?://[^\s\"',]+", stripped):
            candidate_urls.append(match.group(0))

    # Step 2: follow library/publisher links to find actual PDF URLs
    pdf_urls: list[str] = []
    page_urls: list[str] = []
    for url in candidate_urls:
        lower = url.lower()
        if lower.endswith(".pdf") or "/pdf/" in lower or "bitstream" in lower:
            pdf_urls.append(url)
        elif any(domain in lower for domain in ("oapen.org", "archive.org", "jstor.org", "lup.be")):
            page_urls.append(url)

    # Follow library/publisher pages to find actual PDF download links
    visited: set[str] = set()
    pages_to_visit = list(page_urls)
    for _ in range(3):  # max depth
        if not pages_to_visit:
            break
        next_pages: list[str] = []
        for page_url in pages_to_visit:
            if page_url in visited:
                continue
            visited.add(page_url)
            try:
                resp = requests.get(page_url, timeout=15, headers={"User-Agent": USER_AGENT})
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    if href.startswith("/"):
                        parsed_page = urlparse(page_url)
                        href = f"{parsed_page.scheme}://{parsed_page.netloc}{href}"
                    lower_href = href.lower()
                    if ("bitstream" in lower_href or "/pdf" in lower_href) and ".pdf" in lower_href:
                        pdf_urls.append(href.split("?")[0] + "?" + href.split("?")[1] if "?" in href else href)
                    elif "oapen.org/handle" in lower_href and href not in visited:
                        next_pages.append(href)
            except Exception:
                continue
        pages_to_visit = next_pages

    # Step 3: try to query each PDF URL for the answer
    for pdf_url in pdf_urls:
        try:
            excerpts = query_webpage(pdf_url, query, max_snippets=5)
            if excerpts and not excerpts.startswith("Fetch error:") and len(excerpts) > 100:
                return f"Source: {title}\nURL: {pdf_url}\n\n{excerpts}"
        except Exception:
            continue

    # Fallback: search for the title + query terms
    if title:
        try:
            with DDGS() as ddgs:
                results = ddgs.text(f"{title} PDF {query}", max_results=5)
            for result in (results or []):
                url = result.get("href", "").strip()
                if not url:
                    continue
                try:
                    excerpts = query_webpage(url, query, max_snippets=5)
                    if excerpts and not excerpts.startswith("Fetch error:") and len(excerpts) > 100:
                        return f"Source: {result.get('title', '')}\nURL: {url}\n\n{excerpts}"
                except Exception:
                    continue
        except Exception:
            pass

    return f"Found '{title}' but could not extract relevant content for the query."
