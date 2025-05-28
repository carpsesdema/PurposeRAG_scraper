import os
import logging
import hashlib  # For basic hashing if needed, though SmartDeduplicator might have its own
from typing import List, Dict, Optional, Any

import config
from .rag_models import FetchedItem, ParsedItem, NormalizedItem, EnrichedItem, RAGOutputItem
from .fetcher_pool import FetcherPool
from .content_router import ContentRouter
from .chunker import Chunker
from .parser import _clean_snippet_text, extract_code, \
    extract_relevant_links  # Assuming extract_code is the primary general parser

# Attempt to import existing utilities
try:
    from utils.deduplicator import SmartDeduplicator
    from utils.code_categorizer import CodeCategorizer
    from utils.pinescript_categorizer import PinescriptCategorizer

    # from scraper.quality_filter import CodeQualityFilter # If you have a general one
    ADVANCED_UTILS_AVAILABLE = True
except ImportError as e:
    logging.getLogger(config.DEFAULT_LOGGER_NAME).warning(
        f"Could not import one or more advanced utilities (SmartDeduplicator, CodeCategorizer, etc.): {e}. "
        "Pipeline will use basic normalization and enrichment."
    )
    ADVANCED_UTILS_AVAILABLE = False


    # Define dummy classes if advanced utils are not available to prevent runtime errors
    class SmartDeduplicator:
        def __init__(self, logger=None): self.seen_hashes = set()

        def add_snippet(self, code, metadata=None):
            h = hashlib.md5(code.encode()).hexdigest()
            if h in self.seen_hashes: return False
            self.seen_hashes.add(h);
            return True

        def is_duplicate(self, code):  # Ensure this matches expected interface if called directly
            h = hashlib.md5(code.encode()).hexdigest()
            return h in self.seen_hashes, "exact_hash" if h in self.seen_hashes else "unique"


    class CodeCategorizer:
        def __init__(self, logger=None): pass

        def categorize_snippet(self, code, metadata=None): return {'categories': ['general'], 'complexity': 'unknown',
                                                                   'freelance_score': 0, 'client_value': 'N/A',
                                                                   'use_cases': [], 'patterns': [], 'imports': []}


    class PinescriptCategorizer:
        def __init__(self, logger=None): pass

        def categorize_pinescript(self, code, metadata=None): return {'categories': ['pinescript'],
                                                                      'complexity': 'unknown', 'trading_value_score': 0,
                                                                      'script_type': 'unknown', 'client_value': 'N/A',
                                                                      'use_cases': [], 'patterns': [],
                                                                      'pinescript_version': 'unknown'}


class Exporter:
    def __init__(self, logger, output_path: str = "./data/output.jsonl"):
        self.logger = logger
        self.output_path = output_path
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def export(self, rag_output_items: List[RAGOutputItem]):
        self.logger.info(f"Exporting {len(rag_output_items)} RAG items to {self.output_path}")
        try:
            with open(self.output_path, 'a', encoding='utf-8') as f:
                for item in rag_output_items:
                    f.write(item.model_dump_json() + '\n')
            self.logger.info(f"Successfully exported items to {self.output_path}")
        except Exception as e:
            self.logger.error(f"Failed to export items to {self.output_path}: {e}", exc_info=True)


def run_pipeline(query: str, config_path: Optional[str] = None, logger=None, progress_callback=None,
                 initial_content_type: Optional[str] = None):
    if logger is None:
        logger = logging.getLogger(config.DEFAULT_LOGGER_NAME)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            logger.addHandler(ch)
            logger.setLevel(logging.INFO)

    logger.info(f"Pipeline starting for query: '{query}' with initial content type hint: {initial_content_type}")

    # --- Utility for progress updates ---
    total_steps = 8  # Approximate number of major steps
    current_step = 0

    def _progress_update(step_name: str):
        nonlocal current_step
        current_step += 1
        percentage = int((current_step / total_steps) * 100)
        if progress_callback:
            progress_callback(f"{step_name} ({current_step}/{total_steps})", percentage)
        logger.info(f"Pipeline progress: {step_name} ({current_step}/{total_steps}) - {percentage}%")

    # 1. Config Manager (Conceptual - using global config for now)
    _progress_update("Initializing Configuration")
    # In a full implementation, a ConfigManager would load source definitions,
    # selectors, crawl rules, etc., from a YAML/JSON file.
    # For now, seed URLs will be very basic or derived from the query.

    # 2. Initialize Components
    _progress_update("Initializing Components")
    fetcher_pool = FetcherPool(num_workers=getattr(config, 'MAX_CONCURRENT_FETCHERS', 3), logger=logger)
    content_router = ContentRouter(logger=logger)  # Uses extract_code, is_pinescript_code

    # Initialize Normalizer/Deduplicator
    if ADVANCED_UTILS_AVAILABLE and config.SMART_DEDUPLICATION_ENABLED:
        deduplicator = SmartDeduplicator()  # Assuming it has a compatible interface
        logger.info("Using SmartDeduplicator.")
    else:
        deduplicator = SmartDeduplicator(logger=logger)  # Basic hash-based version
        logger.info("Using basic hash-based deduplicator.")

    # Initialize Metadata Enricher components
    py_categorizer = None
    ps_categorizer = None
    if ADVANCED_UTILS_AVAILABLE and config.CODE_CATEGORIZATION_ENABLED:
        py_categorizer = CodeCategorizer()  # Assuming interface like categorize_snippet
        if config.PINESCRIPT_ENABLED:
            ps_categorizer = PinescriptCategorizer()  # Assuming interface like categorize_pinescript
        logger.info("Advanced categorizers initialized.")
    else:
        logger.info("Basic categorization will be used.")

    chunker = Chunker(logger=logger)
    exporter = Exporter(logger=logger, output_path=getattr(config, 'DEFAULT_EXPORT_PATH', "./data/output.jsonl"))

    # --- Generate Tasks for FetcherPool ---
    _progress_update("Preparing Fetch Tasks")
    seed_tasks = []
    if query.startswith("http://") or query.startswith("https://"):
        source_type_guess = "website"  # Basic guess
        if "github.com" in query:
            source_type_guess = "github_url"
        elif "stackoverflow.com" in query:
            source_type_guess = "stackoverflow_url"
        # Add more intelligent source type guessing here based on URL patterns
        seed_tasks.append({'url': query, 'source_type': source_type_guess, 'query_used': query, 'depth': 0})
    else:
        # Placeholder: In a real system, this uses the ConfigManager to get seeds for the query.
        # For tennis example, config might define wikipedia, atptour.com, wtatennis.com etc. as sources.
        # Each source would have rules for how to derive URLs from the query (e.g., search URL template)
        # For "tennis" query:
        seed_tasks.append(
            {'url': f"https://en.wikipedia.org/wiki/Tennis", 'source_type': 'wikipedia', 'query_used': query,
             'depth': 0})
        seed_tasks.append(
            {'url': f"https://www.atptour.com/en/scores/results-archive?year=2023", 'source_type': 'atp_results',
             'query_used': query, 'depth': 0})
        seed_tasks.append(
            {'url': f"https://www.espn.com/tennis/", 'source_type': 'espn_tennis', 'query_used': query, 'depth': 0})
        # More sophisticated: use search engines with the query, then scrape results.
        # from googlesearch import search # Example, would need to install 'google' library
        # try:
        #     for j in search(query + " site:wikipedia.org", num_results=2): # Limit results
        #         seed_tasks.append({'url': j, 'source_type': 'wikipedia_search', 'query_used': query, 'depth': 0})
        # except Exception as e_search:
        #     logger.warning(f"Google search failed: {e_search}")

    for task_info in seed_tasks:
        fetcher_pool.submit_task(task_info['url'], task_info['source_type'], task_info['query_used'])

    # 3. Fetch Stage
    _progress_update(f"Fetching from {len(seed_tasks)} Sources")
    fetched_items: List[FetchedItem] = fetcher_pool.get_results()
    logger.info(f"Fetched {len(fetched_items)} raw items.")
    if not fetched_items:
        logger.warning("No items were fetched. Exiting pipeline.")
        fetcher_pool.shutdown()
        if hasattr(logger, 'enhanced_snippet_data'): logger.enhanced_snippet_data = []
        return []

    # 4. Parse Stage (Content Routing)
    _progress_update("Parsing Content")
    parsed_items: List[ParsedItem] = []
    for item in fetched_items:
        if item.content:
            parsed = content_router.route_and_parse(item)
            if parsed:
                # Attempt to extract more links for crawling if depth allows (conceptual)
                # if item.depth < current_config.get_crawl_depth(item.source_type):
                #     links = extract_relevant_links(item.content, item.source_url, logger=logger)
                #     for link in links: # Submit new tasks to fetcher_pool
                #         fetcher_pool.submit_task(link, item.source_type + "_crawl", item.query_used, depth=item.depth + 1)
                parsed_items.append(parsed)
        else:
            logger.warning(f"Skipping parsing for {item.source_url} due to empty content.")
    logger.info(f"Parsed {len(parsed_items)} items.")

    # 5. Normalize & Deduplicate Stage
    _progress_update("Normalizing & Deduplicating")
    normalized_items: List[NormalizedItem] = []
    for p_item in parsed_items:
        # Basic Normalization
        cleaned_text_content = None
        if p_item.main_text_content:
            cleaned_text_content = _clean_snippet_text(p_item.main_text_content)
            # TODO: Add more normalization like whitespace, boilerplate removal (if HTML)

        cleaned_code_snippets_data = []
        if p_item.code_snippets:
            for code_str in p_item.code_snippets:
                cleaned_code = _clean_snippet_text(code_str)
                # TODO: Language detection per snippet if not already done in ContentRouter/Parser
                lang_for_snippet = p_item.detected_language or 'unknown'
                if cleaned_code:  # Ensure snippet is not empty after cleaning
                    cleaned_code_snippets_data.append({'code': cleaned_code, 'language': lang_for_snippet})

        # Create content string for deduplication
        # SmartDeduplicator might operate on the rawest form of text/code or normalized.
        # Here, we'll use the cleaned versions for deduplication check.
        content_for_dedup_check = cleaned_text_content or ""
        for cs_data in cleaned_code_snippets_data:
            content_for_dedup_check += cs_data['code']

        is_dup = False
        if content_for_dedup_check.strip():  # Only deduplicate if there's actual content
            if ADVANCED_UTILS_AVAILABLE and config.SMART_DEDUPLICATION_ENABLED:
                # SmartDeduplicator's `add_snippet` returns False if it's a duplicate.
                # It might need an ID or the content itself.
                # Let's assume it has `is_duplicate(content_string)` and `add_content(content_string)`
                is_dup, _reason = deduplicator.is_duplicate(content_for_dedup_check)  # Check first
                if not is_dup:
                    deduplicator.add_snippet(content_for_dedup_check)  # Add if not duplicate
            else:  # Basic hash deduplication
                if not deduplicator.add_snippet(content_for_dedup_check):  # add_snippet returns False if duplicate
                    is_dup = True
        else:  # No content to process
            logger.debug(f"Skipping normalization for {p_item.source_url} due to no content after cleaning.")
            continue

        if not is_dup:
            norm_item = NormalizedItem(
                id=p_item.id,
                parsed_item_id=p_item.id,
                source_url=p_item.source_url,
                source_type=p_item.source_type,
                query_used=p_item.query_used,
                title=p_item.title,  # Preserve title
                cleaned_text=cleaned_text_content,
                cleaned_code_snippets=cleaned_code_snippets_data,
                is_duplicate=False  # Handled above
            )
            normalized_items.append(norm_item)
        else:
            logger.info(f"Item {p_item.source_url} was a duplicate and removed after normalization.")

    logger.info(f"Normalized to {len(normalized_items)} unique items.")

    # 6. Metadata Enrichment Stage
    _progress_update("Enriching Metadata")
    enriched_items: List[EnrichedItem] = []
    gui_enhanced_data_accumulator = []  # To populate logger.enhanced_snippet_data

    for n_item in normalized_items:
        # Determine content type for categorization (could be from ParsedItem or re-evaluated)
        item_content_type = initial_content_type  # Start with hint
        if n_item.cleaned_code_snippets and n_item.cleaned_code_snippets[0]['language'] != 'unknown':
            item_content_type = n_item.cleaned_code_snippets[0]['language']
        elif n_item.cleaned_text:  # If primarily text, use general or detect language
            item_content_type = n_item.detected_language or 'text'  # Assuming detected_language was on NormalizedItem

        enrichment_results = {}
        # Use appropriate categorizer
        current_categorizer = None
        if item_content_type == 'pinescript' and ps_categorizer:
            current_categorizer = ps_categorizer
            # For Pinescript, we might primarily enrich its code snippets
            if n_item.cleaned_code_snippets:
                enrichment_results = current_categorizer.categorize_pinescript(n_item.cleaned_code_snippets[0]['code'])
            elif n_item.cleaned_text:  # If Pinescript was detected but only text found (e.g. documentation)
                enrichment_results = current_categorizer.categorize_pinescript(
                    n_item.cleaned_text)  # Treat text as script
        elif item_content_type == 'python' and py_categorizer:
            current_categorizer = py_categorizer
            if n_item.cleaned_code_snippets:
                enrichment_results = current_categorizer.categorize_snippet(n_item.cleaned_code_snippets[0]['code'])
            elif n_item.cleaned_text:  # Python docs, etc.
                # Basic enrichment for pythonic text if no dedicated text categorizer
                enrichment_results = {'categories': ['python_documentation'], 'complexity': 'medium'}
        else:  # General text or other code types
            # Basic enrichment for general text
            if n_item.cleaned_text:
                enrichment_results = {'categories': [n_item.source_type, 'text_content'], 'complexity': 'medium',
                                      'language': n_item.detected_language or 'en'}
            # If code of other types, it will have basic enrichment from below

        # Build EnrichedItem
        enriched_item_payload = {
            'id': n_item.id,
            'normalized_item_id': n_item.id,
            'source_url': n_item.source_url,
            'source_type': n_item.source_type,
            'query_used': n_item.query_used,
            'title': n_item.title or "Untitled",
            'text_content': n_item.cleaned_text,
            'code_items': n_item.cleaned_code_snippets,  # Pass along list of dicts {'code':..., 'language':...}
            'categories': enrichment_results.get('categories', [n_item.source_type]),
            'tags': enrichment_results.get('tags', [n_item.query_used.split(" ")[0]]),  # Simplified
            'quality_score': enrichment_results.get('quality_score', 5.0),  # Placeholder
            'complexity': enrichment_results.get('complexity', 'medium'),
            'freelance_score': enrichment_results.get('freelance_score'),
            'trading_value_score': enrichment_results.get('trading_value_score'),
            'entities': enrichment_results.get('entities', []),
            'use_cases': enrichment_results.get('use_cases', [])
        }

        # For GUI: detailed_enrichment_data_for_gui
        # This should be a list, where each item corresponds to a displayable unit in your GUI
        # (e.g., one code snippet or one block of text).
        item_gui_details = []
        if n_item.cleaned_code_snippets:
            for idx, cs_data in enumerate(n_item.cleaned_code_snippets):
                # Enrich each code snippet if categorizers support per-snippet enrichment
                # For now, use overall enrichment_results and adapt
                snippet_specific_enrichment = enrichment_results if idx == 0 else {
                    'categories': enrichment_results.get('categories', []),
                    'complexity': enrichment_results.get('complexity', 'medium')}  # Simplified

                gui_detail = {
                    'code': cs_data['code'],
                    'language': cs_data.get('language', item_content_type),
                    'score': snippet_specific_enrichment.get('quality_score', 5.0),  # Example
                    'metadata': snippet_specific_enrichment,  # The full dict from categorizer
                    'content_type': cs_data.get('language', item_content_type),  # For GUI filtering/display
                    # Add other fields your GUI expects for each snippet
                    'source_url': str(n_item.source_url),
                    'title': n_item.title or "Code Snippet"
                }
                item_gui_details.append(gui_detail)
        elif n_item.cleaned_text:  # If it's primarily a text item
            gui_detail = {
                'text_content': n_item.cleaned_text,
                'language': item_content_type,
                'score': enrichment_results.get('quality_score', 5.0),
                'metadata': enrichment_results,
                'content_type': item_content_type,
                'source_url': str(n_item.source_url),
                'title': n_item.title or "Text Document"
            }
            item_gui_details.append(gui_detail)

        enriched_item_payload['detailed_enrichment_data_for_gui'] = item_gui_details
        gui_enhanced_data_accumulator.extend(item_gui_details)  # Add to the global list for the logger

        enriched_items.append(EnrichedItem(**enriched_item_payload))

    logger.info(f"Enriched {len(enriched_items)} items.")
    if hasattr(logger, 'enhanced_snippet_data'):
        logger.enhanced_snippet_data = gui_enhanced_data_accumulator
    else:
        logger.warning("Logger does not have 'enhanced_snippet_data' attribute. GUI might not get detailed data.")

    # 7. Chunk & Format Stage
    _progress_update("Chunking & Formatting")
    rag_output_items: List[RAGOutputItem] = []
    for e_item in enriched_items:
        chunks = chunker.chunk_item(e_item)  # Chunker uses DEFAULT_CHUNK_SIZE etc from config
        rag_output_items.extend(chunks)
    logger.info(f"Generated {len(rag_output_items)} RAG-ready items (chunks).")

    # 8. Export Stage
    _progress_update("Exporting Results")
    if rag_output_items:
        exporter.export(rag_output_items)
    else:
        logger.info("No RAG items to export.")

    fetcher_pool.shutdown()
    logger.info("Pipeline finished.")

    # Prepare return for GUI: list of raw code strings for the main display
    gui_raw_display_strings = []
    for gui_detail_item in gui_enhanced_data_accumulator:
        if 'code' in gui_detail_item:
            gui_raw_display_strings.append(gui_detail_item['code'])
        elif 'text_content' in gui_detail_item:  # For text items
            # Truncate long text for main display if necessary
            text = gui_detail_item['text_content']
            gui_raw_display_strings.append(text[:800] + "..." if len(text) > 800 else text)

    if not gui_raw_display_strings and fetched_items:  # Fallback if enrichment produced nothing for display
        logger.warning("No specific display strings from enrichment; using raw fetched content for GUI list.")
        for fi in fetched_items:
            if fi.content:
                gui_raw_display_strings.append(fi.content[:800] + "..." if len(fi.content) > 800 else fi.content)

    return gui_raw_display_strings


# --- Main entry point for GUI ---
def search_and_fetch(query, logger, progress_callback=None, content_type_gui=None):
    logger.info(f"search_and_fetch (main GUI entry) called with query: '{query}', GUI content_type: {content_type_gui}")

    # `content_type_gui` is a hint from the UI.
    # `run_pipeline` can use this, or rely on its own detection/config.
    raw_snippets_for_gui_display = run_pipeline(
        query,
        logger=logger,
        progress_callback=progress_callback,
        initial_content_type=content_type_gui  # Pass the GUI hint
    )

    # `run_pipeline` is now responsible for populating `logger.enhanced_snippet_data`

    if progress_callback: progress_callback("Search complete!", 100)  # Final progress update

    return raw_snippets_for_gui_display


# --- Fallback/Legacy function stubs (if needed by other parts of old code) ---
# These should ideally be removed or refactored if not strictly necessary.
def fetch_stdlib_docs(module_name, logger):
    logger.warning("Legacy fetch_stdlib_docs called directly. Consider using the main pipeline.")
    return [f"# Placeholder: stdlib docs for {module_name} would be fetched by the pipeline."]


def fetch_stackoverflow_snippets(query, logger, top_n=None):
    logger.warning("Legacy fetch_stackoverflow_snippets called directly.")
    return [f"# Placeholder: StackOverflow for {query} would be fetched by the pipeline."], []


def fetch_github_readme_snippets(query, logger, max_repos=None, snippets_per_repo=None):
    logger.warning("Legacy fetch_github_readme_snippets called directly.")
    return [f"# Placeholder: GitHub READMEs for {query} would be fetched by the pipeline."]


def fetch_github_file_snippets(query, logger, max_repos=None, files_per_repo_target=None):
    logger.warning("Legacy fetch_github_file_snippets called directly.")
    return [f"# Placeholder: GitHub files for {query} would be fetched by the pipeline."], []


def detect_content_type(query: str, logger) -> str:  # Kept for potential direct use if needed
    logger.debug(f"Detecting content type for query: {query}")
    # This function is more of a helper now, the main pipeline's ContentRouter
    # and enrichment steps will make more definitive language/type choices.
    query_lower = query.lower()

    # Prioritize Pinescript if keywords are very specific
    if config.PINESCRIPT_ENABLED:
        ps_keywords = config.CONTENT_TYPE_KEYWORDS.get('pinescript', [])
        if any(keyword in query_lower for keyword in ps_keywords):
            logger.info(f"Detected Pinescript for query '{query}' based on keywords.")
            return 'pinescript'

    # Check for Python keywords
    py_keywords = config.CONTENT_TYPE_KEYWORDS.get('python', [])
    if any(keyword in query_lower for keyword in py_keywords):
        logger.info(f"Detected Python for query '{query}' based on keywords.")
        return 'python'

    # Default based on config or a general 'text' or 'html' if no specific code keywords
    logger.info(
        f"Query '{query}' did not strongly match specific code types, defaulting to '{config.DEFAULT_CONTENT_TYPE}'.")
    return config.DEFAULT_CONTENT_TYPE