# scraper/content_router.py
from multiprocessing import get_logger

from bs4 import BeautifulSoup, Tag  # Ensure Tag is imported for type hinting
import trafilatura
import uuid
from typing import Optional, List, Dict, Any, Union  # Ensure Any is imported

from .config_manager import ConfigManager, SourceConfig, CustomFieldConfig  # Import new models
from .rag_models import FetchedItem, ParsedItem, ExtractedLinkInfo
from .parser import (
    extract_formatted_blocks,
    extract_relevant_links,
    parse_pdf_content,
    parse_html_tables,
    parse_html_lists,
    extract_semantic_blocks
)


logger = get_logger(__name__)


class ContentRouter:
    def __init__(self, config_manager: Optional[ConfigManager] = None, logger_instance=None):
        self.config_manager = config_manager
        self.logger = logger_instance if logger_instance else logger
        self.logger.info("ContentRouter initialized.")
        if self.config_manager:
            self.logger.info("ConfigManager available for site-specific parsing rules.")

    def _extract_custom_fields(self, soup: BeautifulSoup, source_config: SourceConfig) -> Dict[str, Any]:
        """
        Extracts custom fields based on the SourceConfig's selector definitions.
        """
        custom_data: Dict[str, Any] = {}
        if not source_config.selectors or not source_config.selectors.custom_fields:
            return custom_data

        self.logger.debug(
            f"Attempting to extract {len(source_config.selectors.custom_fields)} custom fields for source: {source_config.name}")

        for field_config in source_config.selectors.custom_fields:
            field_name = field_config.name
            extracted_values: List[Any] = []

            try:
                elements = soup.select(field_config.selector)  # soup.select works for CSS selectors
                if not elements:
                    self.logger.debug(
                        f"Custom field '{field_name}': Selector '{field_config.selector}' found no elements.")
                    custom_data[field_name] = [] if field_config.is_list else None  # Store as empty list or None
                    continue

                for element in elements:
                    value: Optional[Union[str, Dict[str, str]]] = None  # Allow for more complex values if needed later
                    if field_config.extract_type == "text":
                        value = element.get_text(separator=" ", strip=True)
                    elif field_config.extract_type == "attribute":
                        if field_config.attribute_name:
                            attr_value = element.get(field_config.attribute_name)
                            # If attribute_name is 'style', it returns a CSSStyleDeclaration, convert to string
                            if isinstance(attr_value, list):  # some attributes return lists (e.g. class)
                                value = " ".join(attr_value)
                            else:
                                value = str(attr_value) if attr_value is not None else None
                        else:
                            self.logger.warning(
                                f"Custom field '{field_name}': extract_type is 'attribute' but attribute_name is missing.")
                    elif field_config.extract_type == "html":
                        value = str(element)  # Get the full HTML of the element

                    if value is not None:  # Ensure value was extracted
                        extracted_values.append(value)

                if field_config.is_list:
                    custom_data[field_name] = extracted_values
                    self.logger.debug(
                        f"Custom field '{field_name}' (list): Extracted {len(extracted_values)} values using '{field_config.selector}'.")
                elif extracted_values:  # Not a list, take the first element
                    custom_data[field_name] = extracted_values[0]
                    self.logger.debug(
                        f"Custom field '{field_name}': Extracted '{str(extracted_values[0])[:50]}...' using '{field_config.selector}'.")
                else:  # No values extracted for a single field
                    custom_data[field_name] = None
                    self.logger.debug(
                        f"Custom field '{field_name}': No value extracted using '{field_config.selector}'.")

            except Exception as e:
                self.logger.error(
                    f"Error extracting custom field '{field_name}' with selector '{field_config.selector}': {e}",
                    exc_info=False)
                custom_data[field_name] = [] if field_config.is_list else None  # Default on error

        return custom_data

    def route_and_parse(self, fetched_item: FetchedItem) -> Optional[ParsedItem]:
        self.logger.info(
            f"Routing content for {fetched_item.source_url}, "
            f"Content-Type: {fetched_item.content_type_detected}, Source hint: {fetched_item.source_type}"
        )

        main_text_content: Optional[str] = None
        extracted_structured_blocks: List[Dict[str, any]] = []
        extracted_custom_fields: Dict[str, Any] = {}  # <-- For custom K-V pairs
        links_info: List[ExtractedLinkInfo] = []
        title: Optional[str] = fetched_item.title
        parser_meta = {}

        # Get site-specific configuration if available (for custom field selectors)
        site_specific_config: Optional[SourceConfig] = None
        if self.config_manager:
            site_specific_config = self.config_manager.get_site_config_for_url(str(fetched_item.source_url))
            if site_specific_config:
                self.logger.debug(
                    f"Using site-specific config: {site_specific_config.name} for {fetched_item.source_url}")

        http_content_type = fetched_item.content_type_detected.lower() if fetched_item.content_type_detected else ''
        url_lower = str(fetched_item.source_url).lower()

        if 'application/pdf' in http_content_type or url_lower.endswith(".pdf"):
            # ... (PDF parsing logic remains the same) ...
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
                    extracted_text_trafilatura = trafilatura.extract(
                        html_content_str, include_comments=False, include_tables=False,
                        include_formatting=False, output_format='text'
                    )
                    main_text_content = extracted_text_trafilatura.strip() if extracted_text_trafilatura else None
                    self.logger.debug(
                        f"Trafilatura main text ({len(main_text_content or '')} chars) from {fetched_item.source_url}")

                    soup = BeautifulSoup(html_content_str, 'lxml')

                    if not title:
                        # Try custom title selector first if available
                        if site_specific_config and site_specific_config.selectors and site_specific_config.selectors.title:
                            title_element = soup.select_one(site_specific_config.selectors.title)
                            if title_element: title = title_element.get_text(strip=True)

                        if not title:  # Fallback to standard title extraction
                            title_tag = soup.find('title')
                            if title_tag and title_tag.string:
                                title = title_tag.string.strip()
                            elif soup.h1:
                                title = soup.h1.get_text(strip=True)
                        self.logger.debug(f"Parsed title: '{title}' for {fetched_item.source_url}")

                    links_info = extract_relevant_links(soup, str(fetched_item.source_url))

                    # --- Extract Custom Fields if config is available ---
                    if site_specific_config:
                        extracted_custom_fields = self._extract_custom_fields(soup, site_specific_config)
                        self.logger.info(
                            f"Extracted {len(extracted_custom_fields)} custom key-value fields for {fetched_item.source_url}.")
                    # --- End Custom Field Extraction ---

                    semantic_elements = extract_semantic_blocks(soup, str(fetched_item.source_url))
                    if semantic_elements: extracted_structured_blocks.extend(semantic_elements)

                    extracted_tables = parse_html_tables(soup, str(fetched_item.source_url))
                    if extracted_tables: extracted_structured_blocks.extend(extracted_tables)

                    extracted_lists = parse_html_lists(soup, str(fetched_item.source_url))
                    if extracted_lists: extracted_structured_blocks.extend(extracted_lists)

                    pre_formatted_blocks = extract_formatted_blocks(soup, str(fetched_item.source_url))
                    if pre_formatted_blocks: extracted_structured_blocks.extend(pre_formatted_blocks)

                except Exception as e:
                    self.logger.error(f"Error parsing HTML from {fetched_item.source_url}: {e}", exc_info=True)
                    if not main_text_content: main_text_content = fetched_item.content
            else:
                self.logger.warning(f"HTML identified but no text content for {fetched_item.source_url}")

        # ... (Plain text/Markdown and JSON/XML parsing logic remains the same) ...
        elif any(ct in http_content_type for ct in ['text/plain', 'text/markdown']) or \
                any(url_lower.endswith(ext) for ext in ['.txt', '.md', '.markdown']):
            parser_meta['source_type_used_for_parsing'] = 'text_or_markdown'
            main_text_content = fetched_item.content
            if not title: title = url_lower.split('/')[-1].split('.')[0].replace("_", " ").title()
        elif any(ct in http_content_type for ct in ['application/json', 'application/xml', 'text/xml']):
            parser_meta['source_type_used_for_parsing'] = 'json_or_xml'
            main_text_content = None
            raw_data_content = fetched_item.content
            block_type = "full_content_json" if 'json' in http_content_type else "full_content_xml"
            lang_hint = "json" if 'json' in http_content_type else "xml"
            extracted_structured_blocks.append({"type": block_type, "language": lang_hint, "content": raw_data_content,
                                                "source_url": str(fetched_item.source_url)})
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

        if not main_text_content and not extracted_structured_blocks and not extracted_custom_fields:
            self.logger.warning(f"No parsable content or custom fields found for {fetched_item.source_url}")
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
            custom_fields=extracted_custom_fields,  # <-- Pass the extracted custom fields
            extracted_links=links_info,
            parser_metadata=parser_meta
        )