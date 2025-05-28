# config.py
"""
Centralized configuration for the Modular RAG Content Scraper.
"""

# --- General Application Settings ---
APP_NAME = "ModularRAGScraper"
DEFAULT_LOGGER_NAME = "rag_scraper_logger"
LOG_FILE_PATH = "rag_scraper.log" # Ensure project root is writable or change path
LOG_LEVEL_CONSOLE = "INFO"
LOG_LEVEL_FILE = "DEBUG"

# --- GUI Settings ---
STYLESHEET_PATH = "gui/styles.qss" # Relative to project root
DEFAULT_WINDOW_TITLE = "Modular RAG Content Scraper"
DEFAULT_WINDOW_WIDTH = 900  # As per your GUI file
DEFAULT_WINDOW_HEIGHT = 700 # As per your GUI file

# --- Fetcher Settings (for the backend pipeline) ---
DEFAULT_REQUEST_TIMEOUT = 15
USER_AGENT = f"{APP_NAME}/1.0"
MAX_CONCURRENT_FETCHERS = 3 # Max parallel downloads for the backend
DOCS_CRAWL_RESPECT_ROBOTS_TXT = True # For web politeness

# --- Content Type Settings ---
# Defines what kind of content the backend pipeline might primarily target or how it categorizes.
# The GUI's content_type_combo will need to align with these if used for backend hints.
# For a general scraper, 'html' (for web pages) and 'text' (for extracted content) are key.
# 'python' can remain if scraping code examples from general web pages is a use case.
CONTENT_TYPES = {
    'html': True,        # General web pages
    'python': True,      # For extracting Python code examples from web pages
    'text': True,        # For plain text extraction or future PDF/text files
    # 'pinescript': False, # Disabled as requested
}
DEFAULT_CONTENT_TYPE = 'html' # Default for GUI if 'auto' isn't well-defined for generic content

# --- Language Detection (can be useful for general content) ---
LANGUAGE_DETECTION_ENABLED = True # For backend processing stage

# --- Source Settings (General - for backend TDD alignment) ---
# These are conceptual types for the backend pipeline to understand where content came from.
# The GUI's SEARCH_SOURCES_COUNT is a UI concern for progress steps.
SEARCH_SOURCES_COUNT = 4 # Number of search "phases" shown in GUI progress

# --- Content Processing & Quality (Backend TDD alignment) ---
QUALITY_FILTER_ENABLED = True # If a general quality filter should be applied by the backend
QUALITY_SCORE_WEIGHTS = { # Generic weights, can be adapted
    'generic': { 'min_score': 2, 'text_clarity_weight': 1.5, 'structure_weight': 1.0 }
}
SMART_DEDUPLICATION_ENABLED = True # For backend Deduplicator

# --- Categorization Settings (Backend TDD alignment) ---
CODE_CATEGORIZATION_ENABLED = True # If generic content categorization (e.g., topic modeling) is desired
                                   # Or, if 'python' content type is active, for Python code categorization.
# If 'python' is an active content type:
CATEGORIZATION_SETTINGS = {
    'python': { 'freelance_scoring': False, 'complexity_analysis': True, 'pattern_detection': True },
    'html': { 'topic_modeling': True, 'metadata_extraction': True } # Example for general HTML
}

# --- Export Settings (GUI alignment and Backend TDD) ---
EMBEDDING_RAG_EXPORT_ENABLED = True # For GUI RAG export button
DUAL_LLM_EXPORT = False # Simplified, can be re-enabled if a generic dual LLM makes sense
# Export formats available in the GUI (can be general for text/markdown)
EXPORT_FORMATS_BY_LANGUAGE = { # Renaming to EXPORT_FORMATS_BY_CONTENT_TYPE might be better
    'html': ['jsonl', 'markdown', 'text_bundle'], # Example for general web content
    'python': ['jsonl', 'markdown', 'code_pack'], # Example for Python code
}

# --- UI Enhancements (GUI alignment) ---
SHOW_LANGUAGE_SELECTOR = True # If GUI should show a content type selector
LANGUAGE_SPECIFIC_UI_HINTS = True # If GUI placeholders/tooltips should change
LANGUAGE_DISPLAY_NAMES = { # For GUI content type combo
    'html': 'Web Page (HTML)',
    'python': 'Python Code Snippets',
    'text': 'Plain Text',
    # 'auto': 'Auto-Detect Content' # Auto-detect for generic content might be complex
}

# --- Settings from your GUI config that might still apply generally ---
FREELANCE_MODE = True # If this flag changes UI elements beyond just code, keep it.
                      # Otherwise, if it was purely for Python freelance code, consider removing.
                      # Your GUI uses this for some right-panel visibility.

# --- Web Scraper Specific Settings (for backend TDD modules like Fetcher, Parser) ---
# These are for the core TDD pipeline that would fetch and parse general web content.
STDLIB_DOCS_BASE_URL = "https://docs.python.org/3/library/{module_name}.html" # Example if scraping Python docs
STDLIB_DOCS_TIMEOUT = 15

STACKEXCHANGE_API_BASE_URL = "https://api.stackexchange.com/2.3"
STACKOVERFLOW_SITE_NAME = "stackoverflow" # If SO is a target source for examples/discussions
STACKOVERFLOW_SEARCH_ENDPOINT = "/search/advanced"
STACKOVERFLOW_ANSWERS_ENDPOINT = "/questions/{qid}/answers"
STACKOVERFLOW_SEARCH_MAX_RESULTS = 10
STACKOVERFLOW_SEARCH_TIMEOUT = 20
STACKOVERFLOW_ANSWERS_MAX_PER_QUESTION = 3
STACKOVERFLOW_ANSWERS_TIMEOUT = 15

GITHUB_README_MAX_REPOS = 5 # For scraping documentation/articles from GitHub READMEs
GITHUB_README_SNIPPETS_PER_REPO = 3
GITHUB_FILES_MAX_REPOS = 0 # Disabled direct code file scraping unless specifically for 'python' type
GITHUB_FILES_PER_REPO_TARGET = 0
GITHUB_FILES_CANDIDATE_MULTIPLIER = 2
GITHUB_MAX_FILE_SIZE_KB = 1024 # For READMEs or general text files on GitHub
GITHUB_FILE_DOWNLOAD_TIMEOUT = 20

EXTRACT_WHOLE_SMALL_PY_FILES = False # If 'python' content_type is used, this could apply
MAX_LINES_FOR_WHOLE_FILE_EXTRACTION = 150
DEFAULT_SNIPPET_FILENAME_SLUG_MAX_LENGTH = 60

# --- Chunking Settings for RAG (Backend TDD) ---
DEFAULT_CHUNK_SIZE = 512  # Number of tokens or characters, depending on strategy
CHUNK_OVERLAP = 50
MIN_CHUNK_SIZE = 50 # Discard chunks smaller than this

# --- Normalization Settings (Backend TDD) ---
REMOVE_BOILERPLATE = True # Attempt to remove nav, ads, footers from HTML
NORMALIZE_WHITESPACE = True

# --- Metadata Enrichment (Backend TDD) ---
# Examples for general content
METADATA_ENRICHMENT_FIELDS = {
    'extract_publication_date': True,
    'extract_author': True,
    'generate_keywords': True, # e.g., using TF-IDF or a lightweight model
    'estimate_reading_time': True,
}