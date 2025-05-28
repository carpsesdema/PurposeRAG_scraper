from typing import Optional
from .rag_models import FetchedItem, ParsedItem
from scraper.parser import extract_code as extract_code_html_markdown  # Your existing
from scraper.pinescript_parser import extract_pinescript_code, is_pinescript_code  # Your existing


# from scraper.parser import extract_code_from_ipynb # If you implement it

# Placeholder for future PDF/Text extractors
# from .pdf_parser import PDFParser
# from .text_parser import TextParser

class ContentRouter:
    def __init__(self, logger):
        self.logger = logger
        # self.pdf_parser = PDFParser() # Example
        # self.text_parser = TextParser() # Example

    def route_and_parse(self, fetched_item: FetchedItem) -> Optional[ParsedItem]:
        self.logger.info(
            f"ContentRouter: Routing and parsing {fetched_item.source_url} (Source: {fetched_item.source_type})")
        content = fetched_item.content

        # Prioritize Pinescript if source_type suggests it or content looks like it
        if fetched_item.source_type == 'pinescript_sources' or \
                fetched_item.source_type == 'tradingview' or \
                is_pinescript_code(content):  # Your existing checker
            self.logger.debug(f"Parsing {fetched_item.source_url} as Pinescript.")
            code_snippets = extract_pinescript_code(content)  # Your existing extractor
            return ParsedItem(
                id=fetched_item.id,
                fetched_item_id=fetched_item.id,
                source_url=fetched_item.source_url,
                source_type=fetched_item.source_type,
                query_used=fetched_item.query_used,
                code_snippets=code_snippets,
                detected_language='pinescript'
            )

        # Generic HTML/Markdown code extraction
        # TODO: Add more sophisticated content type detection here (e.g., from HTTP headers)
        # For now, assume HTML/Markdown if not Pinescript
        self.logger.debug(f"Parsing {fetched_item.source_url} as HTML/Markdown for code.")
        code_snippets = extract_code_html_markdown(content)

        # Try to guess language of these snippets if not explicitly set (e.g. Python)
        # This is a simplification; real language detection might be needed per snippet
        lang_guess = 'python' if any(cs for cs in code_snippets) else None

        return ParsedItem(
            id=fetched_item.id,
            fetched_item_id=fetched_item.id,
            source_url=fetched_item.source_url,
            source_type=fetched_item.source_type,
            query_used=fetched_item.query_used,
            main_text_content=content if not code_snippets else None,  # If no code, store main text
            code_snippets=code_snippets,
            detected_language=lang_guess  # Or more specific based on analysis
        )