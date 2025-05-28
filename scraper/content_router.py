# scraper/content_router.py
import uuid
from multiprocessing import get_logger
from typing import Optional, List, Dict
import trafilatura
from bs4 import BeautifulSoup

from .config_manager import ConfigManager
from .parser import (
    extract_formatted_blocks,
    extract_relevant_links,
    parse_pdf_content,
    parse_html_tables,
    parse_html_lists,
    extract_semantic_blocks  # <-- NEW!
)
from .rag_models import FetchedItem, ParsedItem, ExtractedLinkInfo  # Added ExtractedLinkInfo

logger = get_logger(__name__)


class ContentRouter:
    def __init__(self, config_manager: Optional[ConfigManager] = None, logger_instance=None):
        self.config_manager = config_manager
        self.logger = logger_instance if logger_instance else logger
        self.logger.info("ContentRouter initialized.")
        if self.config_manager:
            self.logger.info("ConfigManager available for site-specific parsing rules.")

    def route_and_parse(self, fetched_item: FetchedItem) -> Optional[ParsedItem]:
        self.logger.info(
            f"Routing content for {fetched_item.source_url}, "
            f"Content-Type: {fetched_item.content_type_detected}, Source hint: {fetched_item.source_type}"
        )

        main_text_content: Optional[str] = None
        extracted_structured_blocks: List[Dict[str, any]] = []
        links_info: List[ExtractedLinkInfo] = []  # Store rich link info
        title: Optional[str] = fetched_item.title
        parser_meta = {}

        http_content_type = fetched_item.content_type_detected.lower() if fetched_item.content_type_detected else ''
        url_lower = str(fetched_item.source_url).lower()

        if 'application/pdf' in http_content_type or url_lower.endswith(".pdf"):
            parser_meta['source_type_used_for_parsing'] = 'pdf'
            if fetched_item.content_bytes:
                main_text_content = parse_pdf_content(fetched_item.content_bytes, str(fetched_item.source_url))
                if not title: title = url_lower.split('/')[-1].replace(".pdf", "").replace("_", " ").title()
            else:
                self.logger.warning(f"PDF identified but no content_bytes for {fetched_item.source_url}")

        elif 'html' in http_content_type or \
                any(url_lower.endswith(ext) for ext in ['.html', '.htm']) or \
                (not http_content_type and fetched_item.content and fetched_item.content.strip().startswith('<')):

            parser_meta['source_type_used_for_parsing'] = 'html'
            html_content_str = fetched_item.content

            if html_content_str:
                try:
                    # Main text with Trafilatura
                    extracted_text_trafilatura = trafilatura.extract(
                        html_content_str, include_comments=False, include_tables=False,
                        include_formatting=False,  # Get plain text from trafilatura
                        output_format='text'
                    )
                    main_text_content = extracted_text_trafilatura.strip() if extracted_text_trafilatura else None
                    self.logger.debug(
                        f"Trafilatura main text ({len(main_text_content or '')} chars) from {fetched_item.source_url}")

                    soup = BeautifulSoup(html_content_str, 'lxml')

                    if not title:
                        title_tag = soup.find('title')
                        if title_tag and title_tag.string:
                            title = title_tag.string.strip()
                        elif soup.h1:
                            title = soup.h1.get_text(strip=True)
                        self.logger.debug(f"Parsed title from HTML: '{title}' for {fetched_item.source_url}")

                    links_info = extract_relevant_links(soup, str(fetched_item.source_url))  # Get rich link info

                    # Semantic blocks first, as they might encompass other elements
                    # or provide broader context.
                    semantic_elements = extract_semantic_blocks(soup, str(fetched_item.source_url))
                    if semantic_elements: extracted_structured_blocks.extend(semantic_elements)

                    # Then tables, lists, and pre-formatted blocks
                    # Consider how to avoid double-parsing content if a semantic block also contains these
                    # e.g. an <article> (semantic) might have tables. Current approach will extract both.
                    # This might be fine, as they represent different views/granularities of content.
                    extracted_tables = parse_html_tables(soup, str(fetched_item.source_url))
                    if extracted_tables: extracted_structured_blocks.extend(extracted_tables)

                    extracted_lists = parse_html_lists(soup, str(fetched_item.source_url))
                    if extracted_lists: extracted_structured_blocks.extend(extracted_lists)

                    pre_formatted_blocks = extract_formatted_blocks(soup, str(fetched_item.source_url))
                    if pre_formatted_blocks: extracted_structured_blocks.extend(pre_formatted_blocks)

                except Exception as e:
                    self.logger.error(f"Error parsing HTML from {fetched_item.source_url}: {e}", exc_info=True)
                    if not main_text_content: main_text_content = fetched_item.content  # Fallback
            else:
                self.logger.warning(f"HTML identified but no text content for {fetched_item.source_url}")

        elif any(ct in http_content_type for ct in ['text/plain', 'text/markdown']) or \
                any(url_lower.endswith(ext) for ext in ['.txt', '.md', '.markdown']):

            parser_meta['source_type_used_for_parsing'] = 'text_or_markdown'
            main_text_content = fetched_item.content
            if not title: title = url_lower.split('/')[-1].split('.')[0].replace("_", " ").title()
            # Future: For Markdown, could use a dedicated parser to extract ```code blocks```
            # and add them to extracted_structured_blocks.

        elif any(ct in http_content_type for ct in ['application/json', 'application/xml', 'text/xml']):
            parser_meta['source_type_used_for_parsing'] = 'json_or_xml'
            main_text_content = None
            raw_data_content = fetched_item.content
            block_type = "full_content_json" if 'json' in http_content_type else "full_content_xml"
            lang_hint = "json" if 'json' in http_content_type else "xml"
            extracted_structured_blocks.append({
                "type": block_type, "language": lang_hint,
                "content": raw_data_content, "source_url": str(fetched_item.source_url)
            })
            if not title: title = url_lower.split('/')[-1].split('.')[0].replace("_", " ").title()

        else:
            self.logger.warning(
                f"Unhandled CType '{http_content_type}' for {fetched_item.source_url}. Treating as text.")
            parser_meta['source_type_used_for_parsing'] = 'unknown_fallback_as_text'
            main_text_content = fetched_item.content
            if not main_text_content and fetched_item.content_bytes:
                try:
                    main_text_content = fetched_item.content_bytes.decode('utf-8', errors='ignore')
                except Exception:
                    self.logger.error(f"Fallback decode failed for {fetched_item.source_url}")
            if not title: title = url_lower.split('/')[-1]

        if not main_text_content and not extracted_structured_blocks:
            self.logger.warning(f"No parsable content found for {fetched_item.source_url}")
            return None

        return ParsedItem(
            id=str(uuid.uuid4()),
            fetched_item_id=fetched_item.id,
            source_url=fetched_item.source_url,
            source_type=fetched_item.source_type,
            query_used=fetched_item.query_used,
            title=title.strip() if title else "Untitled Content",
            main_text_content=main_text_content.strip() if main_text_content else None,
            extracted_structured_blocks=extracted_structured_blocks,
            extracted_links=links_info,  # Pass the rich link info
            parser_metadata=parser_meta
        )
