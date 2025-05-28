# storage/saver.py
import os
import logging

# Setup a logger for this module
# If you have a central logger setup (e.g., from utils.logger), consider using that.
# For simplicity, this creates a basic logger if one isn't found.
try:
    from utils.logger import get_logger

    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)
    if not logger.handlers:  # Avoid adding multiple handlers if already configured
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)  # Default to INFO if not configured by utils


def save_snippets(snippets: list[str], output_directory: str, base_filename: str = "raw_content_item"):
    """
    Saves a list of string snippets to individual files in the specified directory.
    Each snippet is saved as a .txt file.

    Args:
        snippets (list[str]): A list of strings, where each string is the content to save.
        output_directory (str): The directory where snippet files will be created.
                                This function will attempt to create it if it doesn't exist.
        base_filename (str): The prefix for the saved files (e.g., "raw_content_item"
                             will result in files like "raw_content_item_0.txt").

    Raises:
        OSError: If the directory cannot be created.
        IOError: If a file cannot be written.
        Exception: For other unexpected errors during saving.
    """
    if not snippets:
        logger.info("No snippets provided to save.")
        return

    logger.info(f"Attempting to save {len(snippets)} snippets to directory: {output_directory}")

    try:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            logger.info(f"Created output directory: {output_directory}")
    except OSError as e:
        logger.error(f"Failed to create output directory {output_directory}: {e}")
        raise  # Re-raise to be caught by the GUI's SaveWorker

    saved_count = 0
    for i, snippet_content in enumerate(snippets):
        # Ensure snippet_content is a string, as it might be None or other types if data is unexpected
        if not isinstance(snippet_content, str):
            logger.warning(f"Snippet at index {i} is not a string (type: {type(snippet_content)}), skipping.")
            continue

        file_path = os.path.join(output_directory, f"{base_filename}_{i}.txt")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(snippet_content)
            logger.debug(f"Successfully saved snippet to {file_path}")
            saved_count += 1
        except IOError as e:
            logger.error(f"Error writing snippet to file {file_path}: {e}")
            raise  # Re-raise to indicate failure to the SaveWorker
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving {file_path}: {e}")
            raise  # Re-raise for unexpected issues

    if saved_count == len(snippets):
        logger.info(f"Successfully saved all {saved_count} snippets to {output_directory}")
    else:
        logger.warning(f"Saved {saved_count} out of {len(snippets)} provided snippets to {output_directory}")


if __name__ == "__main__":
    # Example Usage for testing this module directly
    logger.setLevel(logging.DEBUG)  # Show debug messages for testing

    test_snippets_list = [
        "This is the first piece of raw content.\nIt has a couple of lines.",
        "Just a single line for the second item.",
        "And the third item is here!\n\nWith some extra spacing too.",
        None,  # Test a non-string item
        "Fourth actual string item."
    ]
    test_output_dir = "test_saved_raw_output"

    print(f"--- Test: Saving Snippets ---")
    print(f"Snippets to save: {len(test_snippets_list)}")
    print(f"Target directory: {test_output_dir}")

    try:
        save_snippets(test_snippets_list, test_output_dir, base_filename="test_item")
        print(f"Test completed. Check the '{test_output_dir}' directory for 'test_item_X.txt' files.")
        print(f"You should find 4 files (one non-string item was skipped).")
        # You might want to manually clean up the test_saved_raw_output directory after checking
    except Exception as e:
        print(f"Error during testing: {e}")