import os
import logging
import hashlib
from typing import List, Dict, Optional, Any
import json
import yaml

import config
from .rag_models import FetchedItem, ParsedItem, NormalizedItem, EnrichedItem, RAGOutputItem
from .fetcher_pool import FetcherPool
from .content_router import ContentRouter
from .chunker import Chunker
from .parser import _clean_text_block, extract_formatted_blocks, extract_relevant_links
from .config_manager import ConfigManager, ExportConfig, SourceConfig  # Ensure SourceConfig is available if used

try:
    from duckduckgo_search import DDGS

    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False
    logging.getLogger(config.DEFAULT_LOGGER_NAME).warning(
        "DuckDuckGo search library not found. Autonomous search will be limited.")

try:
    import spacy
    from langdetect import detect as detect_language, LangDetectException

    NLP_LIBS_AVAILABLE = True
except ImportError:
    NLP_LIBS_AVAILABLE = False
    logging.getLogger(config.DEFAULT_LOGGER_NAME).warning(
        "spaCy or langdetect not found. Advanced NLP enrichment will be disabled.")

try:
    from utils.deduplicator import SmartDeduplicator

    ADVANCED_UTILS_AVAILABLE = True
except ImportError as e:
    logging.getLogger(config.DEFAULT_LOGGER_NAME).warning(f"SmartDeduplicator import failed: {e}.")
    ADVANCED_UTILS_AVAILABLE = False


    class SmartDeduplicator:  # Dummy
        def __init__(self, logger=None): self.seen_hashes = set()

        def add_snippet(self, text_content, metadata=None): h = hashlib.md5(
            text_content.encode()).hexdigest(); return not (h in self.seen_hashes or self.seen_hashes.add(h))

        def is_duplicate(self, text_content): h = hashlib.md5(
            text_content.encode()).hexdigest(); return h in self.seen_hashes, "exact_hash" if h in self.seen_hashes else "unique"

NLP_MODEL = None
if NLP_LIBS_AVAILABLE:
    try:
        NLP_MODEL = spacy.load(getattr(config, 'SPACY_MODEL_NAME', 'en_core_web_sm'))
        logging.getLogger(config.DEFAULT_LOGGER_NAME).info(f"spaCy model '{NLP_MODEL.meta['name']}' loaded.")
    except OSError:  # Model not found
        logging.getLogger(config.DEFAULT_LOGGER_NAME).error(
            f"spaCy model '{getattr(config, 'SPACY_MODEL_NAME', 'en_core_web_sm')}' not found. Please download it. NLP features limited.")
        NLP_MODEL = None
    except Exception as e_nlp_load:  # Other loading errors
        logging.getLogger(config.DEFAULT_LOGGER_NAME).error(f"Error loading spaCy model: {e_nlp_load}")
        NLP_MODEL = None


class Exporter:  # Exporter class remains the same as last version
    def __init__(self, logger, default_output_dir: str = "./data"):
        self.logger = logger;
        self.default_output_dir = default_output_dir
        os.makedirs(self.default_output_dir, exist_ok=True)

    def _determine_export_path_and_format(self, first_item: RAGOutputItem, export_cfg: Optional[ExportConfig] = None) -> \
    tuple[str, str]:
        if export_cfg and export_cfg.output_path and export_cfg.format:
            path_str, format_str = export_cfg.output_path, export_cfg.format.lower()
            os.makedirs(os.path.dirname(path_str), exist_ok=True)
            self.logger.info(f"Using configured export: Path='{path_str}', Format='{format_str}'")
        else:
            ts, qs, ss = first_item.timestamp.replace(":", "-").split(".")[0], first_item.query_used.replace(" ",
                                                                                                             "_").replace(
                "/", "_")[:30], first_item.source_type.replace(" ", "_").replace("/", "_")[:20]
            source_dir = os.path.join(self.default_output_dir, ss);
            os.makedirs(source_dir, exist_ok=True)
            path_str, format_str = os.path.join(source_dir, f"{qs}_{ts}_fallback.jsonl"), "jsonl"
            self.logger.warning(
                f"No specific export config for '{first_item.source_type}'. Fallback: Path='{path_str}', Format='{format_str}'")
        return path_str, format_str

    def export_batch(self, batch_items: List[RAGOutputItem], export_cfg: Optional[ExportConfig] = None):
        if not batch_items: self.logger.info("No items in batch to export."); return
        output_path_str, export_format = self._determine_export_path_and_format(batch_items[0], export_cfg)
        self.logger.info(f"Exporting {len(batch_items)} items to {output_path_str} ({export_format})")
        try:
            if export_format == "jsonl":
                with open(output_path_str, 'a', encoding='utf-8') as f:
                    for item in batch_items: f.write(item.model_dump_json() + '\n')
            elif export_format == "markdown":
                with open(output_path_str, 'a', encoding='utf-8') as f:
                    for item in batch_items:
                        meta_yaml = yaml.dump(item.model_dump(exclude={'chunk_text', 'id'}), sort_keys=False,
                                              allow_unicode=True, indent=2)
                        f.write(
                            f"---\n{meta_yaml}---\n\n## Chunk ID: {item.id} ({item.chunk_index}/{item.total_chunks_for_parent_element})\n\n")
                        if item.title: f.write(f"### Original Title: {item.title}\n\n")
                        lang_h = item.language if item.language and item.language not in ['unknown', 'text'] else ''
                        f.write(f"```{lang_h}\n{item.chunk_text}\n```\n\n<hr />\n\n")
            else:
                self.logger.error(f"Unsupported export format '{export_format}'."); return
            self.logger.info(f"Appended {len(batch_items)} items to {output_path_str}")
        except Exception as e:
            self.logger.error(f"Export failed: {e}", exc_info=True)


def run_pipeline(query_or_config_path: str, logger=None, progress_callback=None,
                 initial_content_type_hint: Optional[str] = None):
    if logger is None:
        logger = logging.getLogger(config.DEFAULT_LOGGER_NAME);
        if not logger.handlers: ch = logging.StreamHandler(); ch.setLevel(logging.INFO); logger.addHandler(
            ch); logger.setLevel(logging.INFO)
    total_steps = 8;
    current_step = 0

    def _progress_update(step_name: str):
        nonlocal current_step;
        current_step += 1;
        percentage = int((current_step / total_steps) * 100)
        if progress_callback: progress_callback(f"{step_name} ({current_step}/{total_steps})", percentage)
        logger.info(f"Pipeline progress: {step_name} ({current_step}/{total_steps}) - {percentage}%")

    _progress_update("Initializing Configuration");
    cfg_manager = ConfigManager(logger=logger)
    is_config_file_mode = os.path.exists(query_or_config_path) and query_or_config_path.lower().endswith(
        (".yaml", ".json"))
    domain_query_for_log = query_or_config_path
    if is_config_file_mode:
        if not cfg_manager.load_config(query_or_config_path): logger.error("Config load failed."); return []
        domain_query_for_log = cfg_manager.config.domain_info.get('name', query_or_config_path)
    logger.info(f"Pipeline starting for: '{domain_query_for_log}'")

    _progress_update("Initializing Components")
    fetcher_pool = FetcherPool(num_workers=getattr(config, 'MAX_CONCURRENT_FETCHERS', 3), logger=logger)
    content_router = ContentRouter(logger=logger)
    deduplicator = SmartDeduplicator(
        logger=logger) if ADVANCED_UTILS_AVAILABLE and config.SMART_DEDUPLICATION_ENABLED else SmartDeduplicator(
        logger=logger)
    chunker = Chunker(logger=logger)
    exporter = Exporter(logger=logger, default_output_dir=getattr(config, 'DEFAULT_EXPORT_DIR', "./data"))

    _progress_update("Preparing Fetch Tasks");
    current_run_export_config: Optional[ExportConfig] = None  # Overall export config for this run
    if cfg_manager.config and cfg_manager.config.sources:
        for src_cfg_model in cfg_manager.get_sources():
            if not current_run_export_config: current_run_export_config = src_cfg_model.export_config
            for seed_url in src_cfg_model.seeds: fetcher_pool.submit_task(str(seed_url),
                                                                          src_cfg_model.source_type or src_cfg_model.name,
                                                                          domain_query_for_log)
    elif not is_config_file_mode:
        query_str = query_or_config_path
        default_export_path = os.path.join(exporter.default_output_dir,
                                           f"{query_str.replace(' ', '_')[:30].lower()}_output.jsonl")
        current_run_export_config = ExportConfig(output_path=default_export_path, format="jsonl")
        if query_str.startswith(("http://", "https://")):
            fetcher_pool.submit_task(query_str, "direct_url_query", query_str)
        elif DUCKDUCKGO_AVAILABLE:
            logger.info(f"Using DuckDuckGo for query: '{query_str}'")
            try:
                with DDGS(timeout=10) as ddgs:
                    for r_idx, r in enumerate(
                            ddgs.text(query_str, max_results=getattr(config, 'AUTONOMOUS_SEARCH_MAX_RESULTS', 5))):
                        if r_idx >= getattr(config, 'AUTONOMOUS_SEARCH_MAX_RESULTS', 5): break
                        if r.get('href'): fetcher_pool.submit_task(r['href'], "autonomous_web_search",
                                                                   query_str); logger.info(f"  DDGS found: {r['href']}")
            except Exception as e_ddgs:
                logger.error(f"DuckDuckGo search failed: {e_ddgs}")
        else:
            logger.error("Autonomous search not possible."); fetcher_pool.shutdown(); return []
    else:
        logger.error("No sources. Aborting."); fetcher_pool.shutdown(); return []

    _progress_update(f"Fetching Content");
    fetched_items_all: List[FetchedItem] = fetcher_pool.get_results()
    if not fetched_items_all: logger.warning("No items fetched."); fetcher_pool.shutdown(); return []

    _progress_update("Parsing Content");
    parsed_items_all: List[ParsedItem] = []
    for item in fetched_items_all:
        if item.content:
            parsed = content_router.route_and_parse(item)
            if parsed: parsed_items_all.append(parsed)
    logger.info(f"Parsed {len(parsed_items_all)} items.")

    _progress_update("Normalizing & Deduplicating");
    normalized_items_all: List[NormalizedItem] = []
    for p_item in parsed_items_all:
        cleaned_text_content = _clean_text_block(p_item.main_text_content) if p_item.main_text_content else None

        # Clean the structured blocks
        final_cleaned_structured_blocks = []
        if p_item.extracted_structured_blocks:
            for block_dict in p_item.extracted_structured_blocks:
                cleaned_content = _clean_text_block(block_dict.get('content', ''))
                if cleaned_content:  # Only add if content remains after cleaning
                    final_cleaned_structured_blocks.append({
                        'type': block_dict.get('type', 'unknown_block'),  # Preserve type
                        'content': cleaned_content,
                        # 'language_hint': block_dict.get('language_hint') # Preserve language hint if parser provided it
                    })

        # Content for deduplication check combines main text and content of structured blocks
        content_for_dedup = (cleaned_text_content or "") + "".join(
            block['content'] for block in final_cleaned_structured_blocks)

        is_dup = False
        if content_for_dedup.strip():  # Only deduplicate if there's actual content
            is_dup, _ = deduplicator.is_duplicate(content_for_dedup)
            if not is_dup: deduplicator.add_snippet(content_for_dedup)
        else:
            continue  # Skip if no content after cleaning

        if not is_dup:
            norm_item = NormalizedItem(
                id=p_item.id, parsed_item_id=p_item.id, source_url=p_item.source_url,
                source_type=p_item.source_type, query_used=p_item.query_used, title=p_item.title,
                cleaned_text_content=cleaned_text_content,
                cleaned_structured_blocks=final_cleaned_structured_blocks,  # Store the list of dicts
                language_of_main_text=p_item.detected_language_of_main_text
            )
            normalized_items_all.append(norm_item)
    logger.info(f"Normalized to {len(normalized_items_all)} unique items.")

    _progress_update("Enriching Metadata");
    enriched_items_all: List[EnrichedItem] = []
    gui_enhanced_data_accumulator = []  # For logger.enhanced_snippet_data
    for n_item in normalized_items_all:
        item_language = n_item.language_of_main_text or initial_content_type_hint or 'en'

        # Base enrichment data
        enrichment_data = {
            'categories': [n_item.source_type], 'tags': [], 'overall_entities': [],
            'language_of_primary_text': item_language, 'quality_score': 5.0, 'complexity_score': 3.0
        }

        # NLP Enrichment for primary_text_content
        if n_item.cleaned_text_content and NLP_MODEL and NLP_LIBS_AVAILABLE:
            try:
                doc = NLP_MODEL(n_item.cleaned_text_content[:NLP_MODEL.max_length - 10])
                enrichment_data['overall_entities'] = list(
                    set([{'text': ent.text, 'label_': ent.label_} for ent in doc.ents]))
                enrichment_data['tags'] = list(set([token.lemma_.lower() for token in doc if
                                                    not token.is_stop and not token.is_punct and len(token.lemma_) > 2][
                                                   :10]))
            except Exception as e_spacy:
                logger.warning(f"spaCy processing failed for main text of {n_item.source_url}: {e_spacy}")

            if enrichment_data['language_of_primary_text'] in ['unknown', 'text', None,
                                                               'en'] and NLP_LIBS_AVAILABLE:  # Re-detect if unsure
                try:
                    detected_lang = detect_language(n_item.cleaned_text_content[:1000])
                    enrichment_data['language_of_primary_text'] = detected_lang
                except LangDetectException:
                    enrichment_data['language_of_primary_text'] = 'en'  # Default

        # Prepare enriched_structured_elements
        final_enriched_structured_elements = []
        if n_item.cleaned_structured_blocks:
            for block in n_item.cleaned_structured_blocks:
                # Potentially enrich each block here (e.g., if block['type'] is 'code', detect its language)
                # For now, just pass them through, adding basic structure.
                final_enriched_structured_elements.append({
                    'type': block.get('type', 'unknown_block'),
                    'content': block.get('content', ''),
                    'language': block.get('language_hint'),  # If parser added a hint
                    'entities': []  # Placeholder for block-specific entities
                })

        display_meta_summary = {
            'source': str(n_item.source_url), 'type': n_item.source_type,
            'language': enrichment_data['language_of_primary_text'],
            'categories': enrichment_data['categories'], 'tags': enrichment_data['tags'][:5],
            'entities_count': len(enrichment_data['overall_entities']),
            'structured_blocks_count': len(final_enriched_structured_elements)
        }

        enriched_item = EnrichedItem(
            id=n_item.id, normalized_item_id=n_item.id, source_url=n_item.source_url,
            source_type=n_item.source_type, query_used=n_item.query_used, title=n_item.title or "Untitled Content",
            primary_text_content=n_item.cleaned_text_content,
            enriched_structured_elements=final_enriched_structured_elements,
            categories=enrichment_data['categories'], tags=enrichment_data['tags'],
            overall_entities=enrichment_data['overall_entities'],
            language_of_primary_text=enrichment_data['language_of_primary_text'],
            quality_score=enrichment_data['quality_score'], complexity_score=enrichment_data['complexity_score'],
            displayable_metadata_summary=display_meta_summary
        )
        enriched_items_all.append(enriched_item)
        # For GUI: construct what it needs. This might be a list of dicts, one for main text, one per structured element.
        # For simplicity, just adding the summary. A more complex GUI might want more.
        gui_enhanced_data_accumulator.append(display_meta_summary)

    logger.info(f"Enriched {len(enriched_items_all)} items.")
    if hasattr(logger, 'enhanced_snippet_data'): logger.enhanced_snippet_data = gui_enhanced_data_accumulator

    _progress_update("Chunking & Formatting");
    all_rag_output_items_for_domain: List[RAGOutputItem] = []
    for e_item in enriched_items_all:
        chunks = chunker.chunk_item(e_item)  # Chunker will now need to handle EnrichedItem with new structure
        all_rag_output_items_for_domain.extend(chunks)
    logger.info(f"Generated {len(all_rag_output_items_for_domain)} RAG-ready items (chunks).")

    _progress_update("Exporting Results")
    if all_rag_output_items_for_domain:
        exporter.export_batch(all_rag_output_items_for_domain, export_cfg=current_run_export_config)
    else:
        logger.info("No RAG items to export.")

    fetcher_pool.shutdown();
    logger.info(f"Pipeline finished for: '{domain_query_for_log}'")

    gui_raw_display_strings = []
    for enriched_item_processed in enriched_items_all:
        # Display primary text and a note about structured elements for GUI
        display_str = ""
        if enriched_item_processed.primary_text_content:
            text = enriched_item_processed.primary_text_content
            display_str += text[:600] + "..." if len(text) > 600 else text
        if enriched_item_processed.enriched_structured_elements:
            display_str += f"\n\n--- ({len(enriched_item_processed.enriched_structured_elements)} structured elements found) ---"
        if display_str: gui_raw_display_strings.append(display_str)

    if not gui_raw_display_strings and fetched_items_all:
        for fi in fetched_items_all:
            if fi.content: gui_raw_display_strings.append(
                fi.content[:800] + "..." if len(fi.content) > 800 else fi.content)
    return gui_raw_display_strings


# search_and_fetch and legacy stubs remain the same
def search_and_fetch(query_or_config_path, logger, progress_callback=None, content_type_gui=None):
    logger.info(f"search_and_fetch received: '{query_or_config_path}', GUI content_type hint: {content_type_gui}")
    raw_snippets = run_pipeline(query_or_config_path, logger=logger, progress_callback=progress_callback,
                                initial_content_type_hint=content_type_gui)
    if progress_callback: progress_callback("Search complete!", 100)
    return raw_snippets


def fetch_stdlib_docs(m, l): return []


def fetch_stackoverflow_snippets(q, l, t=None): return [], []


def fetch_github_readme_snippets(q, l, mr=None, spr=None): return []


def fetch_github_file_snippets(q, l, mr=None, fprt=None): return [], []


def detect_content_type(q: str, l) -> str:
    if any(ext in q.lower() for ext in ['.pdf']): return 'pdf'
    if any(ext in q.lower() for ext in ['.html', '.htm', '.xml', '.json']): return 'markup_data'
    return config.DEFAULT_CONTENT_TYPE