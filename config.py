# config.py
"""
Centralized configuration for the Modular RAG Content Scraper.
Designed for domain-agnostic information retrieval.
"""

# --- General Application Settings ---
APP_NAME = "ModularRAGScraper"
DEFAULT_LOGGER_NAME = "rag_scraper_logger"
LOG_FILE_PATH = "rag_scraper.log"  # Consider making this configurable via env var or CLI
LOG_LEVEL_CONSOLE = "INFO"
LOG_LEVEL_FILE = "DEBUG"
DEFAULT_EXPORT_DIR = "./data_exports"  # Changed to avoid conflict if './data' is used for configs

# --- GUI Settings ---
STYLESHEET_PATH = "gui/styles.qss"  # Ensure this path is correct relative to where app runs
DEFAULT_WINDOW_TITLE = "Modular RAG Content Scraper"
DEFAULT_WINDOW_WIDTH = 900
DEFAULT_WINDOW_HEIGHT = 700

# --- Fetcher Settings ---
DEFAULT_REQUEST_TIMEOUT = 15  # seconds
USER_AGENT = f"{APP_NAME}/3.1"  # Incremented version
MAX_CONCURRENT_FETCHERS = 5  # Max threads for fetching
# DOCS_CRAWL_RESPECT_ROBOTS_TXT was removed, rely on YAML config or Fetcher implementation details
AUTONOMOUS_SEARCH_MAX_RESULTS = 15  # For DuckDuckGo searches
DUCKDUCKGO_SEARCH_DELAY = 2.0  # <<< ADDED: Delay in seconds before a DuckDuckGo search

# --- Content Type Settings ---
# These are hints for processing; specific parsers might override or refine.
# The ContentRouter will use these and HTTP headers / URL extensions.
CONTENT_TYPES = {
    'html': True,  # General web pages
    'text': True,  # Plain text documents or extracted content
    'pdf': True,  # ENABLED PDF processing
    'json': True,  # For fetching direct JSON APIs or files
    'xml': True,  # For fetching direct XML feeds or files
    'markdown': True,  # For .md files
}
DEFAULT_CONTENT_TYPE_FOR_GUI = 'html'  # Default for GUI content type selector when 'auto' isn't chosen

# --- Language & NLP Settings ---
LANGUAGE_DETECTION_ENABLED = True  # For enriching items with language metadata
SPACY_MODEL_NAME = 'en_core_web_sm'  # Default spaCy model for NER, etc.

# --- Source & Pipeline Settings ---
SMART_DEDUPLICATION_ENABLED = True  # Enable advanced or basic deduplication
# CONTENT_CATEGORIZATION_ENABLED removed, categories derived from source_type or NLP

# --- Export Settings ---
DEFAULT_EXPORT_FORMATS_SUPPORTED = ['jsonl', 'markdown']  # Formats the Exporter class can handle

# --- UI Enhancements (GUI specific) ---
SHOW_CONTENT_TYPE_SELECTOR_GUI = True  # Whether to show the dropdown in GUI
LANGUAGE_DISPLAY_NAMES_GUI = {  # Maps internal types to user-friendly names in GUI
    'auto': 'Auto-Detect Content',  # Special value for GUI
    'html': 'Web Page (HTML)',
    'text': 'Plain Text Document',
    'pdf': 'PDF Document',
    'json': 'JSON Data',
    'xml': 'XML Data',
    'markdown': 'Markdown Document'
}

# --- Chunking Settings for RAG ---
DEFAULT_CHUNK_SIZE = 1024  # Target characters per chunk
CHUNK_OVERLAP = 100  # Characters of overlap between chunks
MIN_CHUNK_SIZE = 50  # Minimum characters for a valid chunk

# --- Normalization Settings ---
# REMOVE_BOILERPLATE_HTML removed, Trafilatura handles this for HTML
NORMALIZE_WHITESPACE = True  # Applied during text cleaning stages

# --- Metadata Enrichment Fields (conceptual, actual implementation varies) ---
# DEFAULT_METADATA_ENRICHMENT_FIELDS removed, enrichment logic is in the pipeline's enricher step.
# Add to config.py
LOG_LEVEL_CONSOLE = "INFO"
LOG_LEVEL_FILE = "DEBUG"
EXPORT_VALIDATION_ENABLED = True
QUALITY_FILTER_ENABLED = True
PROFESSIONAL_METRICS = True