# scraper/content_router.py
import logging
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

# Fixed logger import - use standard logging instead of multiprocessing.get_logger
logger = logging.getLogger(__name__)


class ContentRouter:
    def __init__(self, config_manager: Optional[ConfigManager] = None, logger_instance=None):
        self.config_manager = config_manager
        self.logger = logger_instance if logger_instance else logger
        self.logger.info("ContentRouter initialized.")
        if self.config_manager:
            self.logger.info("ConfigManager available for site-specific parsing rules.")

    def _extract_single_field_value(self, element_context: Tag, field_config: CustomFieldConfig) -> Optional[Any]:
        """Helper to extract a single value based on field_config, relative to an element_context."""
        try:
            # For sub-selectors, element_context is the parent element (e.g., a <tr> row)
            # For top-level fields, element_context is the main soup object.
            target_elements = element_context.select(field_config.selector)
            if not target_elements:
                return None

            # For non-list sub-fields, usually expect one target or take the first one.
            target_element = target_elements[0]
            value: Optional[Union[str, Dict[str, str]]] = None

            if field_config.extract_type == "text":
                value = target_element.get_text(separator=" ", strip=True)
            elif field_config.extract_type == "attribute":
                if field_config.attribute_name:
                    attr_val = target_element.get(field_config.attribute_name)
                    value = " ".join(attr_val) if isinstance(attr_val, list) else str(
                        attr_val) if attr_val is not None else None
            elif field_config.extract_type == "html":
                value = str(target_element)

            return value
        except Exception as e:
            self.logger.error(
                f"Error extracting sub-field '{field_config.name}' with selector '{field_config.selector}': {e}",
                exc_info=False)
            return None

    def _extract_custom_fields(self, soup: BeautifulSoup, source_config: SourceConfig) -> Dict[str, Any]:
        """
        Extracts custom fields based on the SourceConfig's selector definitions.
        Handles simple fields, lists of simple values, and lists of structured dictionaries.
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
                # Main elements targeted by the current field_config's selector
                main_elements = soup.select(field_config.selector)

                if not main_elements:
                    self.logger.debug(
                        f"Custom field '{field_name}': Selector '{field_config.selector}' found no elements.")
                    # Set default based on whether a list is expected (simple list or structured_list)
                    custom_data[
                        field_name] = [] if field_config.is_list or field_config.extract_type == "structured_list" else None
                    continue

                if field_config.extract_type == "structured_list":
                    if not field_config.sub_selectors:
                        self.logger.warning(
                            f"Custom field '{field_name}' is 'structured_list' but has no sub_selectors. Skipping.")
                        custom_data[field_name] = []
                        continue

                    for item_element in main_elements:  # Each item_element is a row/item
                        item_data: Dict[str, Any] = {}
                        for sub_field_config in field_config.sub_selectors:
                            item_data[sub_field_config.name] = self._extract_single_field_value(item_element,
                                                                                                sub_field_config)
                        extracted_values.append(item_data)

                    custom_data[field_name] = extracted_values  # This is always a list of dicts
                    self.logger.debug(
                        f"Custom field '{field_name}' (structured_list): Extracted {len(extracted_values)} items using '{field_config.selector}'.")

                else:  # Handles 'text', 'attribute', 'html'
                    for element in main_elements:
                        value = self._extract_single_field_value(element, field_config)
                        if value is not None:
                            extracted_values.append(value)

                    if field_config.is_list:
                        custom_data[field_name] = extracted_values
                        self.logger.debug(
                            f"Custom field '{field_name}' (list of simple values): Extracted {len(extracted_values)} values using '{field_config.selector}'.")
                    elif extracted_values:
                        custom_data[field_name] = extracted_values[0]  # Take the first one for non-list simple fields
                        self.logger.debug(
                            f"Custom field '{field_name}': Extracted '{str(extracted_values[0])[:50]}...' using '{field_config.selector}'.")
                    else:
                        custom_data[field_name] = None
                        self.logger.debug(
                            f"Custom field '{field_name}': No value extracted using '{field_config.selector}'.")

            except Exception as e:
                self.logger.error(
                    f"Error processing top-level custom field '{field_name}' with selector '{field_config.selector}': {e}",
                    exc_info=True)  # Changed to True for more detail on these errors
                custom_data[
                    field_name] = [] if field_config.is_list or field_config.extract_type == "structured_list" else None

        return custom_data

    def route_and_parse(self, fetched_item: FetchedItem) -> Optional[ParsedItem]:
        self.logger.info(
            f"Routing content for {fetched_item.source_url}, "
            f"Content-Type: {fetched_item.content_type_detected}, Source hint: {fetched_item.source_type}"
        )

        main_text_content: Optional[str] = None
        extracted_structured_blocks: List[Dict[str, any]] = []
        extracted_custom_fields: Dict[str, Any] = {}
        links_info: List[ExtractedLinkInfo] = []
        title: Optional[str] = fetched_item.title
        parser_meta = {}

        site_specific_config: Optional[SourceConfig] = None
        if self.config_manager:
            site_specific_config = self.config_manager.get_site_config_for_url(str(fetched_item.source_url))
            if site_specific_config:
                self.logger.debug(
                    f"Using site-specific config: {site_specific_config.name} for {fetched_item.source_url}")

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
                    soup = BeautifulSoup(html_content_str, 'lxml')

                    # Extract custom fields first if site-specific config exists
                    # These fields might be the primary data you want.
                    if site_specific_config:
                        extracted_custom_fields = self._extract_custom_fields(soup, site_specific_config)
                        self.logger.info(
                            f"Extracted {len(extracted_custom_fields)} custom key-value fields for {fetched_item.source_url}.")

                    # Title extraction
                    if not title:
                        custom_title_selector = site_specific_config.selectors.title if site_specific_config and site_specific_config.selectors else None
                        if custom_title_selector:
                            title_element = soup.select_one(custom_title_selector)
                            if title_element: title = title_element.get_text(strip=True)
                        if not title:
                            title_tag = soup.find('title')
                            if title_tag and title_tag.string:
                                title = title_tag.string.strip()
                            elif soup.h1:
                                title = soup.h1.get_text(strip=True)
                        self.logger.debug(f"Parsed title: '{title}' for {fetched_item.source_url}")

                    # Main content extraction strategy:
                    # 1. Use site-specific main_content selector if provided.
                    # 2. Fallback to Trafilatura if no site-specific selector or if it fails.
                    # 3. Fallback to cleaned soup.body if Trafilatura also yields little.

                    site_main_content_selector = site_specific_config.selectors.main_content if site_specific_config and site_specific_config.selectors else None
                    if site_main_content_selector:
                        self.logger.debug(
                            f"Attempting site-specific main_content selector: '{site_main_content_selector}'.")
                        main_content_elements = soup.select(site_main_content_selector)
                        if main_content_elements:
                            selected_text_parts = [el.get_text(separator=" ", strip=True) for el in
                                                   main_content_elements]
                            main_text_content = " ".join(filter(None, selected_text_parts)).strip()
                            self.logger.info(
                                f"Used site-specific selector for main_text_content ({len(main_text_content or '')} chars).")

                    if not main_text_content:  # If site-specific selector failed or wasn't provided
                        self.logger.debug("Attempting Trafilatura for main content extraction.")
                        extracted_text_trafilatura = trafilatura.extract(
                            html_content_str, include_comments=False, include_tables=True,
                            include_formatting=False, output_format='txt',
                            favor_precision=True
                        )
                        main_text_content = extracted_text_trafilatura.strip() if extracted_text_trafilatura else None
                        self.logger.debug(
                            f"Trafilatura main text attempt ({len(main_text_content or '')} chars) from {fetched_item.source_url}")

                    if not main_text_content or len(
                            main_text_content) < 150:  # If Trafilatura output is still insufficient
                        self.logger.debug("Trafilatura output insufficient, trying fallback to cleaned soup.body.")
                        if soup.body:
                            # Create a copy of soup for body cleaning to not affect other extractions
                            body_soup = BeautifulSoup(str(soup.body), 'lxml')
                            for unwanted_tag in body_soup.find_all(
                                    ['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
                                unwanted_tag.decompose()
                            main_text_content = body_soup.get_text(separator=" ", strip=True)
                            self.logger.debug(
                                f"Fallback to cleaned soup.body.get_text() for main_text_content ({len(main_text_content or '')} chars).")

                    # Extract other generic structured elements using the original soup
                    links_info = extract_relevant_links(soup, str(fetched_item.source_url))
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
                    if main_text_content and ("<" in main_text_content and ">" in main_text_content):
                        self.logger.warning(
                            f"main_text_content for {fetched_item.source_url} might still contain HTML after error. Clearing.")
                        main_text_content = None
            else:
                self.logger.warning(f"HTML identified but no text content for {fetched_item.source_url}")

        # ... (rest of the PDF, text, JSON/XML, and fallback handling remains the same) ...
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
            # If we have a site_specific_config, custom_fields might have already parsed this.
            # Only add as a generic block if not captured by custom_fields.
            if not extracted_custom_fields or block_type not in [cf.get("type") for cf in
                                                                 extracted_custom_fields.values() if
                                                                 isinstance(cf, dict)]:  # basic check
                extracted_structured_blocks.append(
                    {"type": block_type, "language": lang_hint, "content": raw_data_content,
                     "source_url": str(fetched_item.source_url)})
            if not title: title = url_lower.split('/')[-1].split('.')[0].replace("_", " ").title()

        else:  # Fallback for unknown content types
            self.logger.warning(
                f"Unhandled CType '{http_content_type}' or no content type detected for {fetched_item.source_url}. Attempting robust text extraction.")
            parser_meta['source_type_used_for_parsing'] = 'unknown_fallback_as_text'
            text_from_bytes = None
            if fetched_item.content_bytes and not fetched_item.content:
                try:
                    guessed_encoding = fetched_item.encoding if fetched_item.encoding else 'utf-8'
                    text_from_bytes = fetched_item.content_bytes.decode(guessed_encoding, errors='replace')
                    self.logger.debug(
                        f"Decoded unknown content from bytes using guessed encoding: {guessed_encoding} for {fetched_item.source_url}")
                except Exception as e_decode_unknown:
                    self.logger.error(
                        f"Failed to decode unknown content bytes for {fetched_item.source_url}: {e_decode_unknown}")
            raw_content_for_fallback = text_from_bytes if text_from_bytes is not None else fetched_item.content
            if raw_content_for_fallback:
                content_strip = raw_content_for_fallback.strip()
                if content_strip.startswith("<") and content_strip.endswith(">") and \
                        any(tag in content_strip.lower() for tag in ['<html', '<body', '<xml', '<rss', '<feed']):
                    self.logger.info(
                        f"Fallback: Content for {fetched_item.source_url} appears to be markup. Attempting to parse and clean.")
                    try:
                        fallback_soup = BeautifulSoup(raw_content_for_fallback, 'lxml')
                        for unwanted_tag in fallback_soup.find_all(
                                ['script', 'style', 'head', 'nav', 'footer', 'aside', 'form']):
                            unwanted_tag.decompose()
                        main_text_content = fallback_soup.get_text(separator=' ', strip=True)
                        self.logger.info(
                            f"Fallback: Stripped tags from unknown markup, yielding {len(main_text_content or '')} chars text for {fetched_item.source_url}.")
                        if not title and fallback_soup.title and fallback_soup.title.string:
                            title = fallback_soup.title.string.strip()
                    except Exception as e_soup_fallback:
                        self.logger.warning(
                            f"Fallback: BeautifulSoup failed on unknown markup content for {fetched_item.source_url}. Using raw content as text. Error: {e_soup_fallback}")
                        main_text_content = raw_content_for_fallback
                else:
                    main_text_content = raw_content_for_fallback
                    self.logger.debug(
                        f"Fallback: Treating unknown content for {fetched_item.source_url} as plain text.")
            else:
                main_text_content = None
            if not title: title = url_lower.split('/')[-1]

        if not main_text_content and not extracted_structured_blocks and not extracted_custom_fields:
            self.logger.warning(
                f"No parsable content, structured blocks, or custom fields found for {fetched_item.source_url}. Skipping item.")
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
            custom_fields=extracted_custom_fields,
            extracted_links=links_info,
            parser_metadata=parser_meta
        )