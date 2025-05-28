# config.py
"""
Centralized configuration for the Modular RAG Content Scraper.
Designed for domain-agnostic information retrieval.
"""

# --- General Application Settings ---
APP_NAME = "ModularRAGScraper"
DEFAULT_LOGGER_NAME = "rag_scraper_logger"
LOG_FILE_PATH = "rag_scraper.log"
LOG_LEVEL_CONSOLE = "INFO"
LOG_LEVEL_FILE = "DEBUG"
DEFAULT_EXPORT_DIR = "./data"

# --- GUI Settings ---
STYLESHEET_PATH = "gui/styles.qss"
DEFAULT_WINDOW_TITLE = "Modular RAG Content Scraper"
DEFAULT_WINDOW_WIDTH = 900
DEFAULT_WINDOW_HEIGHT = 700

# --- Fetcher Settings ---
DEFAULT_REQUEST_TIMEOUT = 15
USER_AGENT = f"{APP_NAME}/3.0" # Version up for pure generic focus
MAX_CONCURRENT_FETCHERS = 5
DOCS_CRAWL_RESPECT_ROBOTS_TXT = True
AUTONOMOUS_SEARCH_MAX_RESULTS = 5

# --- Content Type Settings ---
# These are hints for processing; specific parsers might override.
CONTENT_TYPES = {
    'html': True,        # General web pages are primary
    'text': True,        # For plain text extraction or extracted content
    'pdf': False,        # Placeholder for PDF processing (needs implementation)
    # Add other generic types like 'xml', 'json' if needed for direct fetching
}
DEFAULT_CONTENT_TYPE = 'html' # Default for GUI or when type is ambiguous

# --- Language & NLP Settings ---
LANGUAGE_DETECTION_ENABLED = True
SPACY_MODEL_NAME = 'en_core_web_sm'

# --- Source & Pipeline Settings ---
SMART_DEDUPLICATION_ENABLED = True # General deduplication for text
CONTENT_CATEGORIZATION_ENABLED = True # For general text content

# --- Export Settings ---
DEFAULT_EXPORT_FORMATS = ['jsonl', 'markdown']

# --- UI Enhancements ---
SHOW_CONTENT_TYPE_SELECTOR_GUI = True
LANGUAGE_DISPLAY_NAMES_GUI = {
    'html': 'Web Page (HTML)',
    'text': 'Plain Text Document',
    'pdf': 'PDF Document',
    'auto': 'Auto-Detect Content'
}

# --- Chunking Settings for RAG ---
DEFAULT_CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 50

# --- Normalization Settings ---
REMOVE_BOILERPLATE_HTML = True
NORMALIZE_WHITESPACE = True

# --- Metadata Enrichment Fields ---
DEFAULT_METADATA_ENRICHMENT_FIELDS = {
    'extract_publication_date': True,
    'extract_author': True,
    'generate_keywords_from_text': True,
    'estimate_reading_time': True,
    'perform_ner': True,
    'detect_language': True,
}