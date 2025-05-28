# storage/saver.py
import os
import logging
import json
import re
from typing import List, Dict, Any
from scraper.rag_models import EnrichedItem  # Import the EnrichedItem model

# Logger setup
try:
    from utils.logger import get_logger

    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


def _sanitize_filename(name: str, max_length: int = 100) -> str:
    """Sanitizes a string to be a valid filename component."""
    if not name:
        return "untitled"
    # Remove problematic characters
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    # Replace multiple spaces/underscores with a single underscore
    name = re.sub(r'[\s_]+', '_', name.strip())
    # Truncate to max_length
    return name[:max_length]


def _get_file_extension_for_element(element: Dict[str, Any]) -> str:
    """Determines a file extension based on element type or language hint."""
    element_type = element.get('type', 'unknown').lower()
    language = element.get('language', '').lower()  # Language from formatted_text_block or figure

    if 'table_markdown' in element_type:
        return ".md"
    if 'list' in element_type:  # e.g., html_ul_list, html_ol_list
        return ".md"  # Lists are also saved as markdown-like text

    if element_type == 'formatted_text_block':
        if language and language not in ['plaintext', 'text', 'unknown', '']:
            # Basic mapping, can be expanded
            lang_ext_map = {
                'python': '.py', 'javascript': '.js', 'json': '.json',
                'xml': '.xml', 'html': '.html', 'css': '.css',
                'yaml': '.yaml', 'markdown': '.md', 'sql': '.sql',
                'java': '.java', 'csharp': '.cs', 'bash': '.sh', 'shell': '.sh'
            }
            return lang_ext_map.get(language, ".txt")  # Default to .txt if lang not in map
        return ".txt"  # Plaintext formatted blocks

    if element_type == 'semantic_figure_with_caption':
        # The content is text, so .txt is appropriate.
        # If images were saved, this would be different.
        return ".txt"

    # For other semantic blocks or unknown types
    return ".txt"


def save_enriched_items_to_disk(
        enriched_items: List[EnrichedItem],
        base_output_directory: str,
        query_source_name: str = "general_scrape"  # Used to create a top-level folder for this batch
):
    """
    Saves a list of EnrichedItem objects to disk in a structured format.
    Each EnrichedItem gets its own subdirectory.

    Args:
        enriched_items (List[EnrichedItem]): The list of items to save.
        base_output_directory (str): The user-selected base directory.
        query_source_name (str): A name for this scrape job, used for a subfolder.
    """
    if not enriched_items:
        logger.info("No enriched items provided to save.")
        return

    # Create a top-level directory for this specific save operation (e.g., based on query)
    # Sanitize query_source_name for the directory name
    operation_dirname = _sanitize_filename(query_source_name, max_length=50)
    operation_output_dir = os.path.join(base_output_directory, operation_dirname)

    try:
        os.makedirs(operation_output_dir, exist_ok=True)
        logger.info(f"Saving to operation directory: {operation_output_dir}")
    except OSError as e:
        logger.error(f"Failed to create operation directory {operation_output_dir}: {e}")
        raise

    total_items_saved = 0
    for item_idx, item in enumerate(enriched_items):
        # Create a subdirectory for each EnrichedItem
        item_title_sanitized = _sanitize_filename(item.title if item.title else f"item_{item.id[:8]}", max_length=60)
        item_dir_name = f"{str(item_idx).zfill(3)}_{item_title_sanitized}"
        item_output_path = os.path.join(operation_output_dir, item_dir_name)

        try:
            os.makedirs(item_output_path, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directory for item {item.id} at {item_output_path}: {e}")
            continue  # Skip this item if its directory can't be created

        # 1. Save Metadata
        metadata_filepath = os.path.join(item_output_path, "metadata.json")
        try:
            # Create a dict of metadata to save (can customize what's included)
            metadata_to_save = {
                "id": item.id,
                "source_url": str(item.source_url),  # Convert HttpUrl to string for JSON
                "title": item.title,
                "source_type_hint": item.source_type,
                "query_used": item.query_used,
                "language_primary_text": item.language_of_primary_text,
                "categories": item.categories,
                "tags": item.tags,
                "overall_entities_count": len(item.overall_entities),
                "quality_score": item.quality_score,
                "complexity_score": item.complexity_score,
                "processed_timestamp": datetime.now().isoformat()  # Add save timestamp
            }
            with open(metadata_filepath, 'w', encoding='utf-8') as f_meta:
                json.dump(metadata_to_save, f_meta, indent=4)
            logger.debug(f"Saved metadata for item {item.id} to {metadata_filepath}")
        except Exception as e:
            logger.error(f"Failed to save metadata.json for item {item.id}: {e}")

        # 2. Save Primary Text Content
        if item.primary_text_content:
            primary_text_filepath = os.path.join(item_output_path, "main_content.txt")
            try:
                with open(primary_text_filepath, 'w', encoding='utf-8') as f_text:
                    f_text.write(item.primary_text_content)
                logger.debug(f"Saved primary_text_content for item {item.id} to {primary_text_filepath}")
            except Exception as e:
                logger.error(f"Failed to save main_content.txt for item {item.id}: {e}")

        # 3. Save Enriched Structured Elements
        for elem_idx, element in enumerate(item.enriched_structured_elements):
            element_type_sanitized = _sanitize_filename(element.get('type', 'unknown_element'), max_length=30)
            file_extension = _get_file_extension_for_element(element)

            element_filename = f"element_{str(elem_idx).zfill(2)}_{element_type_sanitized}{file_extension}"
            element_filepath = os.path.join(item_output_path, element_filename)

            content_to_save = ""
            if element.get('type') == 'semantic_figure_with_caption':
                # Combine figure and caption content for saving this type
                fig_content = element.get('figure_content', "")
                cap_content = element.get('caption_content', "")
                content_to_save = f"Figure Content:\n{fig_content}\n\nCaption Content:\n{cap_content}".strip()
            else:
                content_to_save = element.get('content', '')

            if content_to_save:  # Only save if there's actual content for the element
                try:
                    with open(element_filepath, 'w', encoding='utf-8') as f_elem:
                        f_elem.write(str(content_to_save))  # Ensure content is string
                    logger.debug(f"Saved structured element {elem_idx} for item {item.id} to {element_filepath}")
                except Exception as e:
                    logger.error(f"Failed to save structured element {elem_idx} for item {item.id}: {e}")
            else:
                logger.debug(
                    f"Skipped saving empty structured element {elem_idx} (type: {element_type_sanitized}) for item {item.id}")

        total_items_saved += 1
        logger.info(f"Completed saving processed content for item: {item.title or item.id} to {item_output_path}")

    logger.info(
        f"Successfully saved {total_items_saved} out of {len(enriched_items)} items to disk under {operation_output_dir}.")


if __name__ == "__main__":
    # Example Usage (requires EnrichedItem model definition accessible)
    from scraper.rag_models import EnrichedItem, HttpUrl  # For testing
    from datetime import datetime

    logger.setLevel(logging.DEBUG)
    print("--- Testing storage.saver.save_enriched_items_to_disk ---")

    # Create some dummy EnrichedItem objects
    dummy_items = [
        EnrichedItem(
            id="item_001", normalized_item_id="norm_001", source_url=HttpUrl("http://example.com/page1"),
            source_type="blog_post", query_used="test query", title="My First Awesome Blog Post",
            primary_text_content="This is the main content of the first blog post. It's quite interesting.",
            enriched_structured_elements=[
                {"type": "html_table_markdown", "content": "| Header1 | Header2 |\n|---|---|\n| R1C1 | R1C2 |",
                 "caption": "Test Table 1"},
                {"type": "formatted_text_block", "language": "python", "content": "print('Hello World')"}
            ],
            categories=["tech", "python"], tags=["example", "testing"], language_of_primary_text="en",
            displayable_metadata_summary={"url": "http://example.com/page1", "title": "My First Awesome Blog Post"}
        ),
        EnrichedItem(
            id="item_002", normalized_item_id="norm_002", source_url=HttpUrl("http://example.com/page2"),
            source_type="news_article", query_used="another query", title="Important News Update",
            primary_text_content="Latest news: something happened today. It was very newsworthy indeed.",
            enriched_structured_elements=[
                {"type": "html_ul_list", "content": "* Item 1\n* Item 2\n  * Nested Item 2.1", "heading": "Key Points"},
                {"type": "semantic_figure_with_caption", "figure_content": "Text describing an image.",
                 "caption_content": "This is the caption for the figure."}
            ],
            categories=["news"], tags=["update", "daily"], language_of_primary_text="en",
            displayable_metadata_summary={"url": "http://example.com/page2", "title": "Important News Update"}
        )
    ]

    test_base_dir = "test_saved_enriched_output"
    test_query_name = "sample_test_scrape"

    print(f"Items to save: {len(dummy_items)}")
    print(f"Target base directory: {test_base_dir}")
    print(f"Operation sub-directory name: {test_query_name}")

    try:
        save_enriched_items_to_disk(dummy_items, test_base_dir, test_query_name)
        print(
            f"Test completed. Check the '{os.path.join(test_base_dir, _sanitize_filename(test_query_name))}' directory.")
        # You might want to manually clean up this directory after checking
        # import shutil
        # if os.path.exists(os.path.join(test_base_dir, _sanitize_filename(test_query_name))):
        #     shutil.rmtree(os.path.join(test_base_dir, _sanitize_filename(test_query_name)))
        #     print("Cleaned up test directory.")

    except Exception as e:
        print(f"Error during testing save_enriched_items_to_disk: {e}")
        logger.error("Test exception:", exc_info=True)