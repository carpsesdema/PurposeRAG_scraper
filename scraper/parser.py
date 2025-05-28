# scraper/parser.py
import re
from io import StringIO, BytesIO
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, NavigableString, Tag
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from pydantic import HttpUrl  # For validating URLs in ExtractedLinkInfo

from utils.logger import get_logger
from .rag_models import ExtractedLinkInfo  # <-- NEW! Import for rich link info

logger = get_logger(__name__)


# --- Text Cleaning Helper (consistent for various content extractions) ---
def _clean_block_text(text: Optional[str]) -> str:
    """Generic text cleaning for extracted blocks: strips, condenses whitespace."""
    if not text:
        return ""
    cleaned = str(text).strip()
    cleaned = re.sub(r'\s\s+', ' ',
                     cleaned)  # Consolidate multiple whitespace chars (including newlines if not handled)
    cleaned = cleaned.replace("\n", " ")  # Replace newlines with spaces for single line representation if desired
    # For multi-line content, this might be too aggressive.
    # Consider if newlines should be preserved or handled by consumer.
    return cleaned


def _clean_text_for_markdown(text: str) -> str:
    """Cleans text for Markdown table cells, removing excessive newlines and escaping pipes."""
    if not text:
        return " "
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("|", "\\|")
    return text if text else " "


# --- Link Extraction ---
def extract_relevant_links(soup: BeautifulSoup, base_url: str) -> list[ExtractedLinkInfo]:
    """
    Extracts and normalizes links, including anchor text and rel attribute.
    Filters to stay on the same domain by default.
    """
    extracted_links_info = []
    base_parsed_url = urlparse(base_url)

    for a_tag in soup.find_all('a', href=True):
        href_val = a_tag['href']
        if not href_val or href_val.startswith('#') or href_val.startswith('mailto:') or href_val.startswith('tel:'):
            continue

        try:
            full_url = urljoin(base_url, href_val)
            parsed_url_obj = urlparse(full_url)

            # Basic validation for a sensible URL structure
            if not (parsed_url_obj.scheme in ['http', 'https'] and parsed_url_obj.netloc):
                logger.debug(f"Skipping invalid or non-http(s) link: {full_url}")
                continue

            # Convert to Pydantic HttpUrl for validation
            try:
                valid_http_url = HttpUrl(full_url)
            except ValueError:
                logger.debug(f"Skipping link due to Pydantic validation error: {full_url}")
                continue

            # Stay on the same domain/subdomain (optional, can be configured)
            if parsed_url_obj.netloc == base_parsed_url.netloc:
                clean_url_str = valid_http_url  # Already cleaned by urljoin and Pydantic validation somewhat

                anchor_text = _clean_block_text(a_tag.get_text(separator=" ", strip=True))
                rel_attribute = a_tag.get('rel')
                rel_attribute_str = " ".join(rel_attribute) if rel_attribute else None

                link_info = ExtractedLinkInfo(
                    url=clean_url_str,
                    text=anchor_text if anchor_text else None,  # Ensure None if empty
                    rel=rel_attribute_str
                )
                extracted_links_info.append(link_info)
        except Exception as e:
            logger.warning(f"Error processing link '{href_val}' from {base_url}: {e}", exc_info=False)

    logger.info(f"Extracted {len(extracted_links_info)} relevant links with details from {base_url}")
    return extracted_links_info


# --- Semantic Block Extraction ---
SEMANTIC_TAGS_TO_EXTRACT = {
    'article': {'name': 'semantic_article', 'extract_children': False},
    # Usually main content, Trafilatura might get it
    'section': {'name': 'semantic_section', 'extract_children': True},  # Sections can be distinct parts
    'aside': {'name': 'semantic_aside', 'extract_children': True},  # Sidebars, related content
    'nav': {'name': 'semantic_navigation', 'extract_children': True},  # Navigation menus (often list of links)
    'header': {'name': 'semantic_header', 'extract_children': True},  # Page or section header
    'footer': {'name': 'semantic_footer', 'extract_children': True},  # Page or section footer
    'figure': {'name': 'semantic_figure_with_caption', 'extract_children': False}
    # Special handling for figure + figcaption
}


def extract_semantic_blocks(soup: BeautifulSoup, source_url: str) -> list[dict]:
    """
    Extracts content from specified HTML5 semantic tags.
    Tries to pair <figure> with <figcaption>.
    """
    semantic_blocks_data = []

    for tag_name, config in SEMANTIC_TAGS_TO_EXTRACT.items():
        found_tags = soup.find_all(tag_name)

        for idx, tag_content in enumerate(found_tags):
            # Avoid double-counting if a tag is nested within another semantic tag we're already processing (e.g. section in article)
            # This is a simple check; more robust might involve tracking seen elements.
            parent_semantic = False
            for p_tag_name in SEMANTIC_TAGS_TO_EXTRACT.keys():
                if tag_content.find_parent(
                        p_tag_name) and p_tag_name != tag_name:  # if it has a different semantic parent
                    # However, if config.extract_children is False for parent, maybe we should extract this child?
                    # This logic can get complex. For now, simple skip if it has any semantic parent.
                    # A better approach might be to process top-down or only select tags that are not children of other selected semantic tags.
                    # For now, let's extract if it's not a child of THE SAME type of semantic tag (e.g. section within section is fine)
                    # This check isn't perfect.
                    pass

            block_info = {
                "type": config['name'],
                "source_url": source_url,
                "element_index": idx,  # Index within this type of semantic tag
                "tag_name": tag_name
            }

            if tag_name == 'figure':
                figure_text_parts = []
                # Get all text not in figcaption
                for child in tag_content.children:
                    if child.name == 'figcaption':
                        continue
                    if isinstance(child, NavigableString):
                        figure_text_parts.append(str(child).strip())
                    elif isinstance(child, Tag):
                        figure_text_parts.append(child.get_text(separator=" ", strip=True))

                block_info["figure_content"] = _clean_block_text(" ".join(filter(None, figure_text_parts)))

                figcaption_tag = tag_content.find('figcaption')
                if figcaption_tag:
                    block_info["caption_content"] = _clean_block_text(
                        figcaption_tag.get_text(separator=" ", strip=True))
                else:
                    block_info["caption_content"] = None

                # Only add if there's some content
                if block_info["figure_content"] or block_info["caption_content"]:
                    semantic_blocks_data.append(block_info)
                    logger.debug(
                        f"Extracted {config['name']} '{idx}' (Figure: {block_info['figure_content'][:30]}..., Caption: {block_info['caption_content'][:30]}...) from {source_url}")

            else:  # For other semantic tags
                # Simple text extraction for now. Could be made more sophisticated.
                # Example: For 'nav', might specifically extract links.
                content_text = _clean_block_text(tag_content.get_text(separator=" ", strip=True))
                if content_text:  # Only add if there's actual text content
                    block_info["content"] = content_text
                    semantic_blocks_data.append(block_info)
                    logger.debug(
                        f"Extracted {config['name']} '{idx}' (Content: {content_text[:50]}...) from {source_url}")

    logger.info(f"Extracted {len(semantic_blocks_data)} semantic blocks from {source_url}")
    return semantic_blocks_data


# --- Table, List, Pre-formatted Block, PDF Parsers (ASSUMED TO BE THE SAME AS PREVIOUS VERSION) ---
def parse_html_tables(soup: BeautifulSoup, source_url: str) -> list[dict]:
    # ... (Same as your last version) ...
    tables_data = []
    for table_idx, table_tag in enumerate(soup.find_all('table')):
        markdown_table = ""
        headers = []
        header_row_tag = table_tag.find('thead') or table_tag
        header_cells = header_row_tag.find_all('th')
        if header_cells:
            headers = [_clean_text_for_markdown(th.get_text(separator=" ", strip=True)) for th in header_cells]
            markdown_table += "| " + " | ".join(headers) + " |\n"
            markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        body_rows_tag = table_tag.find('tbody') or table_tag
        rows = body_rows_tag.find_all('tr', recursive=False)
        if not rows and not headers: rows = table_tag.find_all('tr')
        has_data_rows = False
        for row_tag in rows:
            if headers and row_tag.find('th'):
                is_header_row = all(cell.name == 'th' for cell in row_tag.find_all(['td', 'th']))
                if is_header_row: continue
            cells = row_tag.find_all(['td', 'th'], recursive=False)
            if not cells: continue
            row_data = [_clean_text_for_markdown(cell.get_text(separator=" ", strip=True)) for cell in cells]
            if not headers and any(cell_text.strip() for cell_text in row_data):
                headers = row_data
                markdown_table += "| " + " | ".join(headers) + " |\n"
                markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                continue
            if headers and len(row_data) == len(headers):
                markdown_table += "| " + " | ".join(row_data) + " |\n"
                has_data_rows = True
            elif not headers and row_data:
                markdown_table += "| " + " | ".join(row_data) + " |\n"
                has_data_rows = True
        if markdown_table and has_data_rows:
            caption_tag = table_tag.find('caption')
            caption = caption_tag.get_text(strip=True) if caption_tag else None
            table_info = {"type": "html_table_markdown", "content": markdown_table.strip(), "source_url": source_url,
                          "element_index": table_idx}
            if caption: table_info["caption"] = caption
            tables_data.append(table_info)
    logger.info(f"Extracted {len(tables_data)} tables as Markdown from {source_url}")
    return tables_data


def _list_item_to_text(li_tag: Tag, list_type_char: str, depth: int) -> str:
    # ... (Same as your last version) ...
    prefix = "  " * depth + list_type_char + " "
    item_text = ""
    for content_part in li_tag.contents:
        if isinstance(content_part, NavigableString):
            item_text += content_part.strip() + " "
        elif isinstance(content_part, Tag):
            if content_part.name in ['ul', 'ol']:
                nested_list_char = "*" if content_part.name == 'ul' else "1."
                item_text += "\n" + _parse_single_list(content_part, nested_list_char, depth + 1)
            else:
                item_text += content_part.get_text(separator=" ", strip=True) + " "
    return prefix + item_text.strip().replace('\n', '\n' + "  " * (depth + 1))


def _parse_single_list(list_tag: Tag, list_type_char: str, depth: int) -> str:
    # ... (Same as your last version) ...
    list_items_text = []
    for item_idx, li_tag in enumerate(list_tag.find_all('li', recursive=False)):
        char_to_use = f"{item_idx + 1}." if list_type_char == "1." else list_type_char
        list_items_text.append(_list_item_to_text(li_tag, char_to_use, depth))
    return "\n".join(list_items_text)


def parse_html_lists(soup: BeautifulSoup, source_url: str) -> list[dict]:
    # ... (Same as your last version) ...
    lists_data = []
    all_lists = soup.find_all(['ul', 'ol'])
    top_level_lists = [lst for lst in all_lists if not lst.find_parent(['ul', 'ol'])]
    for list_idx, list_tag in enumerate(top_level_lists):
        list_type_char = "*" if list_tag.name == 'ul' else "1."
        heading_text = None
        prev_sibling = list_tag.find_previous_sibling()
        if prev_sibling and prev_sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            heading_text = prev_sibling.get_text(strip=True)
        parsed_list_text = _parse_single_list(list_tag, list_type_char, depth=0)
        if parsed_list_text:
            list_info = {"type": f"html_{list_tag.name}_list", "content": parsed_list_text.strip(),
                         "source_url": source_url, "element_index": list_idx}
            if heading_text: list_info["heading"] = heading_text
            lists_data.append(list_info)
    logger.info(f"Extracted {len(lists_data)} top-level lists from {source_url}")
    return lists_data


def extract_formatted_blocks(soup: BeautifulSoup, source_url: str) -> list[dict]:
    # ... (Same as your last version) ...
    formatted_blocks = []
    for pre_tag in soup.find_all('pre'):
        block_text = pre_tag.get_text(separator="\n", strip=True)
        lang_class = pre_tag.get('class', [])
        language = "plaintext"
        for cls in lang_class:
            if cls.startswith('language-'):
                language = cls.replace('language-', ''); break
            elif cls.startswith('lang-'):
                language = cls.replace('lang-', ''); break
            elif cls in ['python', 'javascript', 'java', 'csharp', 'sql', 'html', 'css', 'xml', 'json', 'yaml',
                         'markdown', 'bash', 'shell']:
                language = cls; break
        if language == "plaintext" and block_text:
            if block_text.strip().startswith('{') and block_text.strip().endswith('}'):
                language = "json"
            elif block_text.strip().startswith('<') and block_text.strip().endswith('>'):
                language = "xml"
            elif "def " in block_text or "import " in block_text:
                language = "python"
            elif "function(" in block_text or "const " in block_text or "let " in block_text:
                language = "javascript"
        if block_text:
            formatted_blocks.append(
                {"type": "formatted_text_block", "language": language, "content": block_text, "source_url": source_url})
    logger.debug(f"Extracted {len(formatted_blocks)} formatted blocks from {source_url}")
    return formatted_blocks


def parse_pdf_content(pdf_content_bytes: bytes, source_url: str = "PDF source") -> str:
    # ... (Same as your last version) ...
    logger.info(f"Attempting to parse PDF content from {source_url}")
    if not pdf_content_bytes: logger.warning(f"No content bytes for PDF: {source_url}"); return ""
    try:
        output_string = StringIO();
        laparams = LAParams()
        extract_text_to_fp(BytesIO(pdf_content_bytes), output_string, laparams=laparams, output_type='text',
                           codec='utf-8')
        text = output_string.getvalue()
        logger.info(f"Extracted {len(text)} chars from PDF: {source_url}");
        return text
    except Exception as e:
        logger.error(f"PDF parsing error {source_url}: {e}", exc_info=True); return ""
