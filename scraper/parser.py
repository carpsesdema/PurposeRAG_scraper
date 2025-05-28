# scraper/parser.py
import re
import logging
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup, NavigableString, Tag
from io import StringIO, BytesIO
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams

from .rag_models import ExtractedLinkInfo  # Import for rich link info
from pydantic import HttpUrl, BaseModel  # Ensure BaseModel is imported if HttpUrl needs it for validation context
from typing import Optional, List, Dict, Any, Tuple  # Ensure all necessary typing imports are present

# Fixed logger import - use standard logging instead of multiprocessing.get_logger
logger = logging.getLogger(__name__)


# --- Text Cleaning Helper (consistent for various content extractions) ---
def _clean_block_text(text: Optional[str]) -> str:
    """Generic text cleaning for extracted blocks: strips, condenses whitespace, replaces newlines with spaces."""
    if not text:
        return ""
    cleaned = str(text).strip()
    # Consolidate multiple whitespace characters (including newlines, tabs, etc.) into a single space
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Note: Replacing all newlines with spaces might not be ideal for all content types.
    # If preserving line breaks within a block is important for certain outputs (like code blocks),
    # this function might need to be used more selectively or have an option to preserve newlines.
    # For general text content being fed into NLP or displayed in single lines, this is usually fine.
    return cleaned


def _clean_text_for_markdown(text: str) -> str:
    """Cleans text for Markdown table cells, removing newlines and escaping pipes."""
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
    extracted_links_info: List[ExtractedLinkInfo] = []  # Explicit type
    base_parsed_url = urlparse(base_url)

    for a_tag in soup.find_all('a', href=True):
        href_val = a_tag['href']
        if not href_val or href_val.startswith('#') or href_val.startswith('mailto:') or href_val.startswith('tel:'):
            continue

        try:
            full_url_str = urljoin(base_url, href_val)
            parsed_url_obj = urlparse(full_url_str)

            if not (parsed_url_obj.scheme in ['http', 'https'] and parsed_url_obj.netloc):
                logger.debug(f"Skipping invalid or non-http(s) link: {full_url_str}")
                continue

            try:
                # Validate and create HttpUrl instance
                valid_http_url = HttpUrl(full_url_str)
            except ValueError as e_val:
                logger.debug(f"Skipping link due to Pydantic HttpUrl validation error for '{full_url_str}': {e_val}")
                continue

            if parsed_url_obj.netloc == base_parsed_url.netloc:
                anchor_text = _clean_block_text(a_tag.get_text(separator=" ", strip=True))  # USE _clean_block_text
                rel_attribute_list = a_tag.get('rel')  # rel can be a list of strings
                rel_attribute_str = " ".join(rel_attribute_list) if rel_attribute_list else None

                link_info = ExtractedLinkInfo(
                    url=valid_http_url,  # Use the validated HttpUrl object
                    text=anchor_text if anchor_text else None,
                    rel=rel_attribute_str
                )
                extracted_links_info.append(link_info)
        except Exception as e:
            logger.warning(f"Error processing link '{href_val}' from {base_url}: {e}", exc_info=False)

    logger.info(f"Extracted {len(extracted_links_info)} relevant links with details from {base_url}")
    return extracted_links_info


# --- Semantic Block Extraction ---
SEMANTIC_TAGS_TO_EXTRACT = {
    'article': {'name': 'semantic_article'},
    'section': {'name': 'semantic_section'},
    'aside': {'name': 'semantic_aside'},
    'nav': {'name': 'semantic_navigation'},
    'header': {'name': 'semantic_header'},
    'footer': {'name': 'semantic_footer'},
    'figure': {'name': 'semantic_figure_with_caption'}
}


def extract_semantic_blocks(soup: BeautifulSoup, source_url: str) -> list[dict]:
    """
    Extracts content from specified HTML5 semantic tags.
    Tries to pair <figure> with <figcaption>.
    Avoids extracting content from tags nested within another already targeted semantic tag.
    """
    semantic_blocks_data = []

    # Get all candidate tags first to manage nesting
    all_found_tags_with_name: List[Tuple[str, Tag]] = []
    for tag_name_key in SEMANTIC_TAGS_TO_EXTRACT.keys():
        for found_tag_instance in soup.find_all(tag_name_key):
            all_found_tags_with_name.append((tag_name_key, found_tag_instance))

    # Filter out tags that are children of other found semantic tags
    # This helps get more distinct top-level semantic blocks
    top_level_semantic_tags: List[Tuple[str, Tag]] = []
    processed_tag_objects = set()

    for tag_name, tag_instance in all_found_tags_with_name:
        if tag_instance in processed_tag_objects:
            continue

        is_nested_in_another_semantic = False
        for p_name, p_instance in all_found_tags_with_name:
            if tag_instance == p_instance:  # Don't compare with self
                continue
            if p_instance.find(lambda t: t == tag_instance):  # Check if tag_instance is a descendant of p_instance
                # If tag_instance is inside p_instance, it's nested.
                # We only want to process the outermost semantic tag in such cases.
                # This logic might need refinement based on desired behavior for overlapping sections.
                # Current: if it's inside *any* other semantic tag we found, consider it handled by the parent.
                is_nested_in_another_semantic = True
                break

        if not is_nested_in_another_semantic:
            top_level_semantic_tags.append((tag_name, tag_instance))
            # Add all children of this top-level tag to processed_tag_objects to avoid re-processing them
            # if they happen to also be semantic tags (e.g., a <section> inside an <article>).
            # This assumes we only want the content of the outermost semantic tag if they are of different types.
            # For same types (section in section), BeautifulSoup's find_all usually handles them as distinct.
            for desc_tag in tag_instance.find_all(list(SEMANTIC_TAGS_TO_EXTRACT.keys())):
                processed_tag_objects.add(desc_tag)
        processed_tag_objects.add(tag_instance)

    for idx, (tag_name_key, tag_content_instance) in enumerate(top_level_semantic_tags):
        config = SEMANTIC_TAGS_TO_EXTRACT[tag_name_key]
        block_info = {
            "type": config['name'],
            "source_url": source_url,
            "element_index": idx,
            "tag_name": tag_name_key
        }

        if tag_name_key == 'figure':
            figure_text_parts = []
            for child in tag_content_instance.children:
                if child.name == 'figcaption': continue
                if isinstance(child, NavigableString):
                    figure_text_parts.append(str(child).strip())
                elif isinstance(child, Tag):
                    figure_text_parts.append(child.get_text(separator=" ", strip=True))

            block_info["figure_content"] = _clean_block_text(
                " ".join(filter(None, figure_text_parts)))  # USE _clean_block_text

            figcaption_tag = tag_content_instance.find('figcaption')
            block_info["caption_content"] = _clean_block_text(
                figcaption_tag.get_text(separator=" ", strip=True)) if figcaption_tag else None  # USE _clean_block_text

            if block_info["figure_content"] or block_info["caption_content"]:
                semantic_blocks_data.append(block_info)
        else:
            content_text = _clean_block_text(
                tag_content_instance.get_text(separator=" ", strip=True))  # USE _clean_block_text
            if content_text:
                block_info["content"] = content_text
                semantic_blocks_data.append(block_info)

    logger.info(f"Extracted {len(semantic_blocks_data)} top-level semantic blocks from {source_url}")
    return semantic_blocks_data


# --- Table, List, Pre-formatted Block, PDF Parsers ---
def parse_html_tables(soup: BeautifulSoup, source_url: str) -> list[dict]:
    tables_data = []
    for table_idx, table_tag in enumerate(soup.find_all('table')):
        markdown_table = ""
        headers = []
        # Prioritize thead for headers
        thead_tag = table_tag.find('thead')
        if thead_tag:
            header_cells_in_thead = thead_tag.find_all('th')
            if header_cells_in_thead:
                headers = [_clean_text_for_markdown(th.get_text(separator=" ", strip=True)) for th in
                           header_cells_in_thead]

        # Fallback or complement with th in the table directly if no thead headers or incomplete
        if not headers:
            first_tr = table_tag.find('tr')
            if first_tr:
                header_cells_in_tr = first_tr.find_all('th')
                if header_cells_in_tr:
                    headers = [_clean_text_for_markdown(th.get_text(separator=" ", strip=True)) for th in
                               header_cells_in_tr]

        if headers:
            markdown_table += "| " + " | ".join(headers) + " |\n"
            markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

        body_rows_tag = table_tag.find('tbody') if table_tag.find('tbody') else table_tag

        has_data_rows = False
        for row_tag in body_rows_tag.find_all('tr', recursive=False if table_tag.find('tbody') else True):
            # Skip if this row was already used for headers from <th> tags
            if headers and all(cell.name == 'th' for cell in row_tag.find_all(['td', 'th'], recursive=False)):
                # Check if this row's content matches the already extracted headers to avoid duplication
                current_row_th_texts = [_clean_text_for_markdown(th.get_text(separator=" ", strip=True)) for th in
                                        row_tag.find_all('th', recursive=False)]
                if current_row_th_texts == headers:
                    continue

            cells = row_tag.find_all(['td', 'th'], recursive=False)
            if not cells: continue

            row_data = [_clean_text_for_markdown(cell.get_text(separator=" ", strip=True)) for cell in cells]

            if not headers and any(cell_text.strip() for cell_text in row_data):  # First suitable row as header
                headers = row_data
                markdown_table = "| " + " | ".join(headers) + " |\n"  # Reset markdown_table
                markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                continue

            if headers:  # Ensure row matches header count, or adapt (e.g. rowspan/colspan not handled)
                # Simple alignment: pad shorter rows, truncate longer ones
                if len(row_data) < len(headers):
                    row_data.extend([" "] * (len(headers) - len(row_data)))
                elif len(row_data) > len(headers):
                    row_data = row_data[:len(headers)]
                markdown_table += "| " + " | ".join(row_data) + " |\n"
                has_data_rows = True
            elif row_data:  # No headers, just list rows (less structured)
                markdown_table += "| " + " | ".join(row_data) + " |\n"
                has_data_rows = True

        if markdown_table and (has_data_rows or not headers):  # Add if table has data or is headerless with rows
            caption_tag = table_tag.find('caption')
            caption = _clean_block_text(
                caption_tag.get_text(strip=True)) if caption_tag else None  # USE _clean_block_text

            table_info = {"type": "html_table_markdown", "content": markdown_table.strip(), "source_url": source_url,
                          "element_index": table_idx}
            if caption: table_info["caption"] = caption
            tables_data.append(table_info)
    logger.info(f"Extracted {len(tables_data)} tables as Markdown from {source_url}")
    return tables_data


def _list_item_to_text(li_tag: Tag, list_type_char: str, depth: int, is_ordered_parent: bool, item_index: int) -> str:
    """Converts an <li> item to a text line, handling nested lists."""
    actual_char = f"{item_index + 1}." if is_ordered_parent else list_type_char
    prefix = "  " * depth + actual_char + " "

    item_text_parts = []
    for content_part in li_tag.children:  # Use .children to get direct content
        if isinstance(content_part, NavigableString):
            stripped_string = content_part.strip()
            if stripped_string: item_text_parts.append(stripped_string)
        elif isinstance(content_part, Tag):
            if content_part.name in ['ul', 'ol']:
                # Ensure nested list starts on a new line relative to its parent item's text
                nested_parsed_text = _parse_single_list(content_part, depth + 1)
                if nested_parsed_text:  # Add newline only if nested list has content
                    item_text_parts.append("\n" + nested_parsed_text)
            else:  # Other tags within li
                item_text_parts.append(content_part.get_text(separator=" ", strip=True))

    # Join parts, clean up, and handle line continuations
    full_item_text = " ".join(filter(None, item_text_parts))
    full_item_text = re.sub(r'\s+', ' ', full_item_text).strip()  # Consolidate whitespace

    # Indent lines if the item text itself becomes multi-line due to nested lists
    indented_continuation = "\n" + "  " * (depth + len(actual_char) + 1)  # Indent continuation lines more
    full_item_text = full_item_text.replace("\n", indented_continuation)

    return prefix + full_item_text


def _parse_single_list(list_tag: Tag, depth: int) -> str:
    """Helper to parse a single <ul> or <ol> tag recursively."""
    list_items_text = []
    is_ordered = list_tag.name == 'ol'
    list_type_char = "1." if is_ordered else "*"  # This char is mainly for the top-level, index handles ordered items

    for item_idx, li_tag in enumerate(list_tag.find_all('li', recursive=False)):
        list_items_text.append(_list_item_to_text(li_tag, list_type_char, depth, is_ordered, item_idx))
    return "\n".join(list_items_text)


def parse_html_lists(soup: BeautifulSoup, source_url: str) -> list[dict]:
    lists_data = []

    all_lists = soup.find_all(['ul', 'ol'])
    # Filter for lists that are not nested within another list's <li> tag
    # This helps to get distinct lists rather than re-processing sub-lists.
    top_level_lists = []
    for lst in all_lists:
        parent_li = lst.find_parent('li')
        is_top_level = True
        if parent_li:
            # Check if this parent <li> belongs to a list we've already considered (or will consider)
            # This is tricky. A simpler heuristic: if its parent is another ul/ol, it's nested.
            # We want lists whose direct parent is NOT ul/ol, or whose parent ul/ol is not in all_lists (root lists).
            # The _parse_single_list will handle recursion.
            if parent_li.find_parent(['ul', 'ol']):  # If the parent li is itself within a list
                # This means lst is a nested list. It will be handled by recursion in _parse_single_list.
                continue
        top_level_lists.append(lst)

    for list_idx, list_tag in enumerate(top_level_lists):
        heading_text = None
        prev_sibling = list_tag.find_previous_sibling()
        if prev_sibling and prev_sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div']:
            # Check if the text seems like a heading (e.g., short, ends with colon, or is a heading tag)
            temp_heading_text = _clean_block_text(prev_sibling.get_text(strip=True))  # USE _clean_block_text
            if prev_sibling.name.startswith('h') or (len(temp_heading_text) < 100 and temp_heading_text.endswith(':')):
                heading_text = temp_heading_text

        parsed_list_text = _parse_single_list(list_tag, depth=0)

        if parsed_list_text:
            list_info = {
                "type": f"html_{list_tag.name}_list",
                "content": parsed_list_text.strip(),
                "source_url": source_url,
                "element_index": list_idx
            }
            if heading_text: list_info["heading"] = heading_text
            lists_data.append(list_info)

    logger.info(f"Extracted {len(lists_data)} top-level lists from {source_url}")
    return lists_data


def extract_formatted_blocks(soup: BeautifulSoup, source_url: str) -> list[dict]:
    formatted_blocks = []
    for pre_tag in soup.find_all('pre'):
        # Remove any "copy" button/span often found inside <pre> by some highlighters
        for button_or_span in pre_tag.find_all(['button', 'span'], class_=lambda x: x and 'copy' in x.lower()):
            button_or_span.decompose()

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

        # Basic content sniffing if no class hint
        if language == "plaintext" and block_text:
            # Simple sniffs, can be improved
            if re.match(r'^\s*\{.*\}\s*$', block_text, re.DOTALL) or re.match(r'^\s*\[.*\]\s*$', block_text, re.DOTALL):
                language = "json"
            elif re.match(r'^\s*<.+>', block_text, re.DOTALL) and re.search(r'</.+>$', block_text, re.DOTALL):
                language = "xml"  # Could be HTML too
            elif "def " in block_text or "import " in block_text or "class " in block_text:
                language = "python"
            elif "function(" in block_text or "const " in block_text or "let " in block_text or "var " in block_text:
                language = "javascript"

        if block_text:  # Only add if there's actual content
            formatted_blocks.append(
                {"type": "formatted_text_block", "language": language, "content": block_text, "source_url": source_url})
    logger.debug(f"Extracted {len(formatted_blocks)} formatted blocks from {source_url}")
    return formatted_blocks


def parse_pdf_content(pdf_content_bytes: bytes, source_url: str = "PDF source") -> str:
    logger.info(f"Attempting to parse PDF content from {source_url}")
    if not pdf_content_bytes:
        logger.warning(f"No content bytes provided for PDF parsing from {source_url}")
        return ""
    try:
        output_string = StringIO()
        laparams = LAParams()
        extract_text_to_fp(BytesIO(pdf_content_bytes), output_string, laparams=laparams, output_type='text',
                           codec='utf-8')
        text = output_string.getvalue()
        logger.info(f"Successfully extracted {len(text)} characters from PDF: {source_url}")
        return text
    except Exception as e:
        logger.error(f"PDF parsing error for {source_url}: {e}", exc_info=True)
        return ""