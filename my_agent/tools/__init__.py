from .calculator import calculator
from .chess_engine import analyze_chess_position, read_chess_board
from .describe_image import describe_image
from .read_pdf import query_pdf, read_pdf
from .web_search import count_pdf_pages_with_phrase, extract_tables, fetch_webpage, query_webpage, read_doi, web_research, web_search

__all__ = [
    "calculator",
    "count_pdf_pages_with_phrase",
    "extract_tables",
    "fetch_webpage",
    "query_pdf",
    "query_webpage",
    "read_doi",
    "read_pdf",
    "web_research",
    "web_search",
]
