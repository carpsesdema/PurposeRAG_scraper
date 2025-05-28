import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict  # Added Dict for the return type


def _clean_text_block(text_block: str) -> str:
    """Basic cleaning of common artifacts from extracted text blocks."""
    if not isinstance(text_block, str):
        return ""

    # Remove common "copy code" or UI elements often found near preformatted text
    lines = text_block.splitlines()
    cleaned_lines = []
    for line in lines:
        # Heuristic: if a line is very short and contains these, it might be UI noise
        if len(line.strip()) < 20 and any(ui_cmd in line.lower() for ui_cmd in ['copy code', 'copy', 'raw', 'view']):
            continue
        # Remove common prefixes from console examples if any slip through
        if line.startswith(">>> "):
            cleaned_lines.append(line[4:])
        elif line.startswith("... "):
            cleaned_lines.append(line[4:])
        else:
            cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    # More aggressive cleaning of typical UI text that might be near blocks
    cleaned_text = re.sub(r'\bCopy code\b|\bShow code\b|\bView raw\b', '', cleaned_text, flags=re.IGNORECASE).strip()
    return cleaned_text


def extract_formatted_blocks(html_content: str) -> List[Dict[str, str]]:
    """
    Extracts pre-formatted text blocks (e.g., from <pre>, <code>, or Markdown fences)
    from HTML content. Returns a list of dictionaries, each with 'type' and 'content'.
    """
    extracted_elements: List[Dict[str, str]] = []
    seen_content = set()  # To avoid adding literally identical blocks

    # 1. Markdown Fenced Blocks (e.g., ```text ... ```)
    # This regex captures content within ```, optionally with a language hint (which we'll store as 'block_type_hint')
    fence_pattern = re.compile(r'```(?:([a-zA-Z0-9_+-]*)\n)?(.*?)```', re.DOTALL)
    for match in fence_pattern.finditer(html_content):
        block_type_hint = match.group(1) or "fenced_block"  # e.g., 'python', 'json', or just 'fenced_block'
        content_inside_fence = match.group(2).strip()

        if content_inside_fence:
            cleaned_content = _clean_text_block(content_inside_fence)
            if cleaned_content and cleaned_content not in seen_content:
                extracted_elements.append({'type': block_type_hint, 'content': cleaned_content})
                seen_content.add(cleaned_content)

    # 2. HTML <pre> tags
    # These often contain code, configuration, or other structured text.
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        for pre_tag in soup.find_all("pre"):
            # Heuristic: If <pre> tag's content is already captured by a fenced block, skip.
            # This is tricky because rendering might slightly change content. A simple check:
            pre_text_content = pre_tag.get_text().strip()
            if pre_text_content in seen_content:
                continue

            # Check if the <pre> tag has a <code> tag inside with a class hinting at a language
            code_tag_inside = pre_tag.find("code")
            block_type_hint = "preformatted_text"  # Default
            if code_tag_inside and code_tag_inside.get('class'):
                # e.g., class="language-python" -> "python"
                for css_class in code_tag_inside['class']:
                    if css_class.startswith('language-'):
                        block_type_hint = css_class.replace('language-', '', 1)
                        break
                    elif css_class.startswith('lang-'):
                        block_type_hint = css_class.replace('lang-', '', 1)
                        break

            cleaned_content = _clean_text_block(pre_text_content)
            if cleaned_content and cleaned_content not in seen_content:
                extracted_elements.append({'type': block_type_hint, 'content': cleaned_content})
                seen_content.add(cleaned_content)

        # 3. Standalone <code> tags (not inside <pre>) if they are multi-line or substantial
        for code_tag in soup.find_all("code"):
            if code_tag.find_parent("pre"):  # Already handled
                continue

            text_content = code_tag.get_text().strip()
            # Heuristic: if it's multi-line or fairly long and contains non-alphanumeric typical of formatted text
            if ('\n' in text_content or len(text_content) > 40) and \
                    re.search(r'[{}();<>/=:\'"\[\]]', text_content):  # Check for common code/config symbols

                cleaned_content = _clean_text_block(text_content)
                if cleaned_content and cleaned_content not in seen_content:
                    block_type_hint = "inline_block"
                    if code_tag.get('class'):  # Try to get a hint from class
                        for css_class in code_tag['class']:
                            if css_class.startswith('language-') or css_class.startswith('lang-'):
                                block_type_hint = css_class.split('-', 1)
                                break
                    extracted_elements.append({'type': block_type_hint, 'content': cleaned_content})
                    seen_content.add(cleaned_content)

    except Exception as e_bs4:
        # If BS4 fails, we might have already gotten content from regex fences
        # Log this error for debugging.
        # print(f"BeautifulSoup parsing error in extract_formatted_blocks: {e_bs4}") # Or use a logger
        pass

    return extracted_elements


def extract_relevant_links(html_content: str, base_url: str, allowed_domains: List[str] = None, logger=None) -> List[
    str]:
    """
    Extracts relevant hyperlinks from HTML content.
    (This function remains largely the same, as it's generally useful)
    """
    links = set()
    if not html_content or not base_url:
        return []

    try:
        soup = BeautifulSoup(html_content, "html.parser")
        parsed_base_url = urlparse(base_url)
        base_domain = parsed_base_url.netloc

        effective_allowed_domains = set(allowed_domains) if allowed_domains else set()
        effective_allowed_domains.add(base_domain)

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            try:
                absolute_url = urljoin(base_url, href)
                parsed_url = urlparse(absolute_url)

                if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
                    continue

                if parsed_url.netloc in effective_allowed_domains:
                    links.add(absolute_url.split('#'))

            except Exception as e_link:
                if logger:
                    logger.debug(f"Could not process link '{href}' from base '{base_url}': {e_link}")
                continue
        return list(links)
    except Exception as e_html:
        if logger:
            logger.error(f"Error parsing HTML for links (base URL: {base_url}): {e_html}")
        return []

# Removed extract_code_from_ipynb as it's code-specific and you have tools for that.
# If you need a generic JSON content extractor for .ipynb files (e.g., to get all text from markdown cells),
# we could add a different function for that.