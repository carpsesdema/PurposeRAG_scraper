from typing import Optional, List, Dict
from .rag_models import FetchedItem, ParsedItem  # Updated model names
from scraper.parser import extract_formatted_blocks  # Renamed from extract_code
import config  # For logging name or other general configs

try:
    import trafilatura

    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    # Consider logging this unavailability once, perhaps in searcher.py
    # logging.getLogger(config.DEFAULT_LOGGER_NAME).warning("Trafilatura not available. Main text extraction will be basic.")


class ContentRouter:
    def __init__(self, logger):
        self.logger = logger
        # self.pdf_parser = PDFParser() # Would be initialized if PDF support is active

    def route_and_parse(self, fetched_item: FetchedItem) -> Optional[ParsedItem]:
        self.logger.info(
            f"ContentRouter: Routing/parsing {fetched_item.source_url} (Source Type Hint: {fetched_item.source_type}, Detected HTTP Content-Type: {fetched_item.content_type_detected})")

        raw_content = fetched_item.content
        title: Optional[str] = None
        main_text_content: Optional[str] = None
        # This will now store generic structured blocks, not specifically "code"
        extracted_blocks: List[Dict[str, str]] = []
        detected_lang: Optional[str] = None

        # Attempt to determine a more specific content type if not already clear
        effective_content_type = "html"  # Default assumption
        if fetched_item.content_type_detected:
            if 'pdf' in fetched_item.content_type_detected.lower():
                effective_content_type = 'pdf'
            elif 'xml' in fetched_item.content_type_detected.lower():
                effective_content_type = 'xml'  # Placeholder for future XML parser
            elif 'json' in fetched_item.content_type_detected.lower():
                effective_content_type = 'json'  # Placeholder for future JSON parser
            # text/plain often means we should try HTML parsing anyway, as it might be poorly headered HTML
            elif 'html' in fetched_item.content_type_detected.lower() or 'text/plain' in fetched_item.content_type_detected.lower():
                effective_content_type = 'html'

        # --- Routing based on effective_content_type ---
        if effective_content_type == 'pdf':
            self.logger.debug(f"PDF parsing requested for {fetched_item.source_url} (not fully implemented).")
            # main_text_content = self.pdf_parser.extract_text(fetched_item.content_bytes_if_any)
            main_text_content = "PDF parsing placeholder text."
            # title = self.pdf_parser.extract_title()
            self.logger.warning(f"PDF parsing for {fetched_item.source_url} is a placeholder.")

        elif effective_content_type == 'html':
            self.logger.debug(f"Using generic HTML parser for {fetched_item.source_url}.")
            try:  # Extract title
                from bs4 import BeautifulSoup  # Local import is fine here
                soup = BeautifulSoup(raw_content, 'html.parser')
                if soup.title and soup.title.string:
                    title = soup.title.string.strip()
                elif soup.h1:
                    title = soup.h1.get_text().strip()
            except Exception as e_title:
                self.logger.debug(f"Could not extract title for {fetched_item.source_url}: {e_title}")

            if TRAFILATURA_AVAILABLE:  # Extract main text
                extracted_text_trafila = trafilatura.extract(raw_content, include_comments=False, include_tables=True,
                                                             fav_recall=True, output_format='text')
                if extracted_text_trafila:
                    main_text_content = extracted_text_trafila
                else:
                    self.logger.debug(
                        f"Trafilatura extracted no main text from {fetched_item.source_url}. Basic fallback may occur if no structured blocks found.")
            else:
                self.logger.warning(
                    "Trafilatura not available. Main text extraction from HTML will be very basic (full text minus script/style).")
                if 'soup' not in locals(): soup = BeautifulSoup(raw_content,
                                                                'html.parser')  # Avoid re-parsing if already done for title
                for script_or_style in soup(["script", "style"]): script_or_style.decompose()
                main_text_content = soup.get_text(separator='\n', strip=True)

            # Extract any pre-formatted blocks (could be code, examples, etc.)
            # The parser.extract_formatted_blocks should be generic enough.
            extracted_blocks = extract_formatted_blocks(raw_content)

        else:  # Default for 'text' or other unhandled types
            self.logger.debug(
                f"Treating {fetched_item.source_url} as plain text or unhandled type '{effective_content_type}'.")
            main_text_content = raw_content  # Assume the whole content is the main text
            # Try to extract a title from first line if it's plain text
            if not title and main_text_content:
                first_line = main_text_content.split('\n', 1)[0].strip()
                if len(first_line) < 100 and len(first_line) > 5: title = first_line  # Heuristic for title

        if not main_text_content and not extracted_blocks:
            self.logger.warning(
                f"No parsable content (main text or structured blocks) found for {fetched_item.source_url}")
            return None

        # Language detection for main_text_content would typically happen in MetadataEnricher
        # but ParsedItem has a field, so we can make a basic attempt or leave it for later.
        # For now, we'll let the enricher handle it more robustly.

        return ParsedItem(
            id=fetched_item.id,
            fetched_item_id=fetched_item.id,
            source_url=fetched_item.source_url,
            source_type=fetched_item.source_type,
            query_used=fetched_item.query_used,
            title=title,
            main_text_content=main_text_content.strip() if main_text_content else None,
            extracted_structured_blocks=extracted_blocks,
            # detected_language_of_main_text will be populated by MetadataEnricher
        )