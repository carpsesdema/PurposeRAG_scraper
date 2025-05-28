import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


def _clean_snippet_text(snippet_text):
    if not isinstance(snippet_text, str):
        return ""
    lines = snippet_text.splitlines()
    cleaned_lines = []
    for line in lines:
        if line.startswith(">>> "):
            cleaned_lines.append(line[4:])
        elif line.startswith("... "):
            cleaned_lines.append(line[4:])
        else:
            cleaned_lines.append(line)
    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r'\bCopy code\b', '', cleaned_text, flags=re.IGNORECASE).strip()
    return cleaned_text


def extract_code(html_content: str) -> list[str]:
    """
    Extracts code snippets from HTML/Markdown-like content.
    Prioritizes fenced code blocks, then <pre> tags.
    """
    snippets = []

    # Markdown fenced code blocks (Python, generic, or no language specified)
    # ```python\n...\n``` or ```\n...\n```
    fence_patterns = [
        re.compile(r'```(?:python|py)?\n(.*?)\n```', re.DOTALL | re.IGNORECASE),  # Python specific
        re.compile(r'```(?:[a-zA-Z0-9_+-]*\n)?(.*?)```', re.DOTALL)  # Generic or other languages
    ]

    # Process Python-specific blocks first to ensure they are captured accurately
    processed_by_python_fence = set()
    for match in fence_patterns[0].finditer(html_content):
        snippet = match.group(1).strip()
        if snippet:
            cleaned_snippet = _clean_snippet_text(snippet)
            if cleaned_snippet:
                snippets.append(cleaned_snippet)
                # Mark the start position to avoid double processing by generic fence or <pre>
                processed_by_python_fence.add(match.start())

    # Process generic/other language fenced code blocks
    for match in fence_patterns[1].finditer(html_content):
        if match.start() in processed_by_python_fence:  # Skip if already handled by Python-specific
            continue
        snippet = match.group(1).strip()
        if snippet:
            # Basic check if it might be code (e.g., contains common programming symbols)
            if any(char in snippet for char in ['{', '}', ';', '(', ')', '=', '<', '>']):
                cleaned_snippet = _clean_snippet_text(snippet)
                if cleaned_snippet:
                    snippets.append(cleaned_snippet)

    # HTML <pre> and <code> tags
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        for pre_tag in soup.find_all("pre"):
            # Check if this <pre> tag's content might have already been extracted by fenced blocks
            # This is a heuristic: if the <pre> tag is a direct child of a div that often wraps markdown
            is_likely_from_markdown_render = False
            parent = pre_tag.parent
            if parent and parent.name == 'div':
                if any(cls in (parent.get('class', [])) for cls in ['highlight', 'codehilite']):
                    is_likely_from_markdown_render = True

            if is_likely_from_markdown_render and any(pre_tag.get_text().strip() in s for s in snippets):
                continue  # Likely already extracted

            code_elements = pre_tag.find_all("code")
            text_to_extract = ""
            if code_elements:  # Prefer content of <code> inside <pre>
                text_to_extract = "\n".join(code_elem.get_text() for code_elem in code_elements)
            else:
                text_to_extract = pre_tag.get_text()

            cleaned_text = _clean_snippet_text(text_to_extract.strip())
            if cleaned_text:
                snippets.append(cleaned_text)

        # Extract from <code> tags not within <pre> if they seem substantial
        for code_tag in soup.find_all("code"):
            if not code_tag.find_parent("pre"):
                text = code_tag.get_text().strip()
                # Heuristic for substantial inline code: multiple lines or significant length with code-like chars
                if ('\n' in text or len(text) > 30) and any(char in text for char in ['{', '}', ';', '(', ')', '=']):
                    cleaned_text = _clean_snippet_text(text)
                    if cleaned_text:
                        snippets.append(cleaned_text)

    except Exception:  # Broad exception for parsing issues
        pass  # Silently ignore BeautifulSoup errors if HTML is malformed

    return list(dict.fromkeys(snippets))  # Deduplicate


def extract_code_from_ipynb(json_string: str, logger) -> list[str]:
    logger.info("Attempting to extract code from IPYNB content.")
    codes = []
    try:
        import json  # Local import as this function might be specialized
        notebook = json.loads(json_string)
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source_lines = cell.get('source', [])
                if isinstance(source_lines, list):
                    source = "".join(source_lines)
                    if source.strip():
                        codes.append(_clean_snippet_text(source.strip()))
                elif isinstance(source_lines, str):  # Sometimes source is just a string
                    if source_lines.strip():
                        codes.append(_clean_snippet_text(source_lines.strip()))
    except ImportError:
        logger.error("json module not found, cannot parse IPYNB.")
    except Exception as e:
        logger.error(f"Error parsing IPYNB content: {e}")
    return list(dict.fromkeys(codes))


def extract_relevant_links(html_content: str, base_url: str, allowed_domains: list = None, logger=None) -> list[str]:
    links = set()
    if not html_content or not base_url:
        return []

    try:
        soup = BeautifulSoup(html_content, "html.parser")
        parsed_base_url = urlparse(base_url)
        base_domain = parsed_base_url.netloc

        effective_allowed_domains = set(allowed_domains) if allowed_domains else set()
        effective_allowed_domains.add(base_domain)  # Always allow the base domain

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            try:
                absolute_url = urljoin(base_url, href)
                parsed_url = urlparse(absolute_url)

                if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
                    continue

                if parsed_url.netloc in effective_allowed_domains:
                    # Avoid #-fragments unless they are the only thing (which urljoin usually handles)
                    links.add(absolute_url.split('#')[0])

            except Exception as e:
                if logger:
                    logger.debug(f"Could not process link '{href}' from base '{base_url}': {e}")
                continue
        return list(links)
    except Exception as e:
        if logger:
            logger.error(f"Error parsing HTML for links (base URL: {base_url}): {e}")
        return []