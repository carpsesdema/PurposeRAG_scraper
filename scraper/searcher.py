# scraper/searcher.py
import os
import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable, Tuple
import json
import yaml
import re

import config
from utils import logger
from .rag_models import FetchedItem, ParsedItem, NormalizedItem, EnrichedItem, RAGOutputItem, ExtractedLinkInfo
from .fetcher_pool import FetcherPool
from .content_router import ContentRouter
from .chunker import Chunker
# REMOVED: from .parser import _clean_text_block # This import was unused here
from .config_manager import ConfigManager, ExportConfig, SourceConfig

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


    class SmartDeduplicator:
        def __init__(self, logger=None): self.seen_hashes = set(); self.logger = logger or logging.getLogger(
            "DummyDeduplicator")

        def add_snippet(self, text_content, metadata=None):
            h = hashlib.md5(text_content.encode('utf-8', 'replace')).hexdigest()
            if h in self.seen_hashes: return False
            self.seen_hashes.add(h)
            return True

        def is_duplicate(self, text_content):
            h = hashlib.md5(text_content.encode('utf-8', 'replace')).hexdigest()
            return h in self.seen_hashes, "exact_hash" if h in self.seen_hashes else "unique"

NLP_MODEL = None
if NLP_LIBS_AVAILABLE:
    try:
        NLP_MODEL = spacy.load(getattr(config, 'SPACY_MODEL_NAME', 'en_core_web_sm'))
        logging.getLogger(config.DEFAULT_LOGGER_NAME).info(f"spaCy model '{NLP_MODEL.meta['name']}' loaded.")
    except OSError:
        logging.getLogger(config.DEFAULT_LOGGER_NAME).error(
            f"spaCy model '{getattr(config, 'SPACY_MODEL_NAME', 'en_core_web_sm')}' not found. NLP features limited.")
        NLP_MODEL = None
    except Exception as e_nlp_load:
        logging.getLogger(config.DEFAULT_LOGGER_NAME).error(f"Error loading spaCy model: {e_nlp_load}")
        NLP_MODEL = None


class Exporter:
    def __init__(self, logger, default_output_dir: str = "./data_exports"):
        self.logger = logger
        self.default_output_dir = default_output_dir
        os.makedirs(self.default_output_dir, exist_ok=True)

    def _determine_export_path_and_format(self, first_item: RAGOutputItem, export_cfg: Optional[ExportConfig] = None) -> \
            tuple[str, str]:
        if export_cfg and export_cfg.output_path and export_cfg.format:
            path_str, format_str = export_cfg.output_path, export_cfg.format.lower()
            output_parent_dir = os.path.dirname(path_str)
            if output_parent_dir:
                os.makedirs(output_parent_dir, exist_ok=True)
            else:
                path_str = os.path.join(self.default_output_dir, path_str)

            self.logger.info(f"Using configured export: Path='{path_str}', Format='{format_str}'")
        else:
            ts = first_item.timestamp.replace(":", "-").split(".")[0]
            qs_sanitized = re.sub(r'[^\w\-_\. ]', '_', first_item.query_used)
            qs_sanitized = qs_sanitized.replace(' ', '_')[:50]

            ss_sanitized = re.sub(r'[^\w\-_\. ]', '_', first_item.source_type)
            ss_sanitized = ss_sanitized.replace(' ', '_')[:30]

            source_specific_dir_name = f"{qs_sanitized}_{ss_sanitized}".lower()
            final_export_dir = os.path.join(self.default_output_dir, source_specific_dir_name)
            os.makedirs(final_export_dir, exist_ok=True)

            path_str = os.path.join(final_export_dir, f"rag_export_{ts}_fallback.jsonl")
            format_str = "jsonl"
            self.logger.warning(
                f"No specific export config for '{first_item.source_type}'. Fallback: Path='{path_str}', Format='{format_str}'")
        return path_str, format_str

    def export_batch(self, batch_items: List[RAGOutputItem], export_cfg: Optional[ExportConfig] = None):
        if not batch_items: self.logger.info("No items in batch to export."); return

        output_path_str, export_format = self._determine_export_path_and_format(batch_items[0], export_cfg)

        file_output_dir = os.path.dirname(output_path_str)
        if file_output_dir: os.makedirs(file_output_dir, exist_ok=True)

        self.logger.info(f"Exporting {len(batch_items)} RAG items to {output_path_str} (format: {export_format})")
        try:
            file_exists = os.path.exists(output_path_str)
            with open(output_path_str, 'a', encoding='utf-8') as f:
                if export_format == "jsonl":
                    for item in batch_items: f.write(item.model_dump_json() + '\n')
                elif export_format == "markdown":
                    if not file_exists:
                        f.write(f"# RAG Export: {batch_items[0].query_used}\n")
                        f.write(f"Source Type: {batch_items[0].source_type}\n")
                        f.write(
                            f"Exported on: {datetime.now().isoformat()}\n\n<hr />\n\n")  # Corrected import, assuming datetime is available

                    for item in batch_items:
                        meta_dump = item.model_dump(exclude={'chunk_text'})
                        meta_yaml = yaml.dump(meta_dump, sort_keys=False, allow_unicode=True, indent=2,
                                              default_flow_style=False)

                        f.write(f"---\n{meta_yaml}---\n\n")
                        f.write(
                            f"## Chunk ID: {item.id} (Index: {item.chunk_index} / Parent: {item.chunk_parent_type})\n\n")
                        if item.title: f.write(f"### Original Title: {item.title}\n\n")

                        lang_hint_for_md = ""
                        if item.language and item.language.lower() not in ['unknown', 'text', 'en', None]:
                            lang_hint_for_md = item.language.lower()

                        f.write(f"```{lang_hint_for_md}\n{item.chunk_text}\n```\n\n<hr />\n\n")
                else:
                    self.logger.error(f"Unsupported export format '{export_format}'.")
                    return
            self.logger.info(f"Appended {len(batch_items)} RAG items to {output_path_str}")
        except Exception as e:
            self.logger.error(f"RAG Export failed: {e}", exc_info=True)


def _clean_text_for_dedup(text: Optional[str]) -> str:
    """Basic cleaning for deduplication purposes."""
    if not text: return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def run_pipeline(query_or_config_path: str, logger_instance=None,
                 progress_callback: Optional[Callable[[str, int], None]] = None,
                 initial_content_type_hint: Optional[str] = None) -> List[EnrichedItem]:
    logger = logger_instance if logger_instance else logging.getLogger(config.DEFAULT_LOGGER_NAME)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)

    total_steps = 8
    current_step = 0

    def _progress_update(step_name: str):
        nonlocal current_step
        current_step += 1
        percentage = int((current_step / total_steps) * 100)
        if progress_callback: progress_callback(f"{step_name} ({current_step}/{total_steps})", percentage)
        logger.info(f"Pipeline progress: {step_name} ({current_step}/{total_steps}) - {percentage}%")

    _progress_update("Initializing Configuration")
    cfg_manager = ConfigManager(logger=logger)
    is_config_file_mode = os.path.exists(query_or_config_path) and query_or_config_path.lower().endswith(
        (".yaml", ".yml", ".json"))
    domain_query_for_log = query_or_config_path
    if is_config_file_mode:
        if not cfg_manager.load_config(query_or_config_path):
            logger.error("Config load failed. Aborting pipeline.")
            return []
        domain_query_for_log = cfg_manager.config.domain_info.get('name',
                                                                  query_or_config_path) if cfg_manager.config else query_or_config_path
    logger.info(f"Pipeline starting for: '{domain_query_for_log}'")

    _progress_update("Initializing Components")
    fetcher_pool = FetcherPool(num_workers=getattr(config, 'MAX_CONCURRENT_FETCHERS', 3), logger=logger)
    content_router = ContentRouter(config_manager=cfg_manager, logger_instance=logger)

    deduplicator_log_level = logging.DEBUG if config.LOG_LEVEL_FILE.upper() == "DEBUG" else logging.INFO
    dedup_logger = logging.getLogger("DeduplicatorInPipeline")
    dedup_logger.setLevel(deduplicator_log_level)
    if not dedup_logger.handlers: dedup_logger.addHandler(logging.StreamHandler())

    deduplicator = SmartDeduplicator(
        logger=dedup_logger) if ADVANCED_UTILS_AVAILABLE and config.SMART_DEDUPLICATION_ENABLED else SmartDeduplicator(
        logger=dedup_logger)

    chunker = Chunker(logger=logger)
    # Assuming datetime is available for Exporter (it's a standard library)
    from datetime import datetime  # Added for Exporter's markdown part
    exporter = Exporter(logger=logger, default_output_dir=getattr(config, 'DEFAULT_EXPORT_DIR', "./data_exports"))

    _progress_update("Preparing Fetch Tasks")
    current_run_export_config: Optional[ExportConfig] = None
    tasks_to_fetch: List[Tuple[str, str, str, Optional[str]]] = []

    if cfg_manager.config and cfg_manager.config.sources:
        for src_cfg_model in cfg_manager.get_sources():
            if not current_run_export_config and src_cfg_model.export_config:
                current_run_export_config = src_cfg_model.export_config
            for seed_url in src_cfg_model.seeds:
                tasks_to_fetch.append(
                    (str(seed_url), src_cfg_model.source_type or src_cfg_model.name, domain_query_for_log, None))
    elif not is_config_file_mode:
        query_str = query_or_config_path
        qs_sanitized_adhoc = re.sub(r'[^\w\-_\. ]', '_', query_str).replace(' ', '_')[:50]
        adhoc_export_dir = os.path.join(exporter.default_output_dir, f"adhoc_{qs_sanitized_adhoc}".lower())
        os.makedirs(adhoc_export_dir, exist_ok=True)
        default_export_path = os.path.join(adhoc_export_dir, "rag_export.jsonl")
        current_run_export_config = ExportConfig(output_path=default_export_path, format="jsonl")

        if query_str.startswith(("http://", "https://")):
            tasks_to_fetch.append((query_str, "direct_url_query", query_str, None))
        elif DUCKDUCKGO_AVAILABLE:
            logger.info(f"Using DuckDuckGo for query: '{query_str}'")
            try:
                with DDGS(timeout=10) as ddgs:
                    ddg_results = ddgs.text(query_str, max_results=getattr(config, 'AUTONOMOUS_SEARCH_MAX_RESULTS', 5))
                    for r_idx, r_dict in enumerate(ddg_results):
                        if r_dict.get('href'):
                            tasks_to_fetch.append(
                                (r_dict['href'], "autonomous_web_search", query_str, r_dict.get('title')))
                            logger.info(f"  DDGS found: {r_dict['href']} (Title: {r_dict.get('title')})")
            except Exception as e_ddgs:
                logger.error(f"DuckDuckGo search failed: {e_ddgs}", exc_info=True)
        else:
            logger.error("Autonomous search not possible (DuckDuckGo unavailable). Aborting.")
            fetcher_pool.shutdown()
            return []
    else:
        logger.error("No sources defined in config or query provided. Aborting.")
        fetcher_pool.shutdown()
        return []

    for url, s_type, q_used, item_title in tasks_to_fetch:
        fetcher_pool.submit_task(url, s_type, q_used, item_title)

    _progress_update(f"Fetching Content ({len(tasks_to_fetch)} URLs)")
    fetched_items_all: List[FetchedItem] = fetcher_pool.get_results()
    if not fetched_items_all: logger.warning("No items fetched."); fetcher_pool.shutdown(); return []

    _progress_update("Parsing Content")
    parsed_items_all: List[ParsedItem] = []
    for item_fetched in fetched_items_all:
        if item_fetched.content_bytes or item_fetched.content:
            parsed = content_router.route_and_parse(item_fetched)
            if parsed: parsed_items_all.append(parsed)
    logger.info(f"Parsed {len(parsed_items_all)} items out of {len(fetched_items_all)} fetched.")

    _progress_update("Normalizing & Deduplicating")
    normalized_items_all: List[NormalizedItem] = []
    for p_item in parsed_items_all:
        full_text_for_dedup_parts = []
        if p_item.main_text_content:
            full_text_for_dedup_parts.append(_clean_text_for_dedup(p_item.main_text_content))

        cleaned_structured_blocks_for_norm = []
        for block_dict in p_item.extracted_structured_blocks:
            content_to_clean = block_dict.get('content', '')
            if block_dict.get('type') == 'semantic_figure_with_caption':
                figure_c = block_dict.get('figure_content', '')
                caption_c = block_dict.get('caption_content', '')
                content_to_clean = f"{figure_c} {caption_c}".strip()

            cleaned_block_content = _clean_text_for_dedup(content_to_clean)
            if cleaned_block_content:
                full_text_for_dedup_parts.append(cleaned_block_content)
            cleaned_structured_blocks_for_norm.append(block_dict.copy())

        full_content_signature = " ".join(filter(None, full_text_for_dedup_parts)).strip()

        is_dup = False
        if full_content_signature:
            is_dup, _ = deduplicator.is_duplicate(full_content_signature)
            if not is_dup: deduplicator.add_snippet(full_content_signature)
        else:
            logger.debug(f"Skipping deduplication for item with no content: {p_item.source_url}")
            continue

        if not is_dup:
            norm_item = NormalizedItem(
                id=p_item.id, parsed_item_id=p_item.id, source_url=p_item.source_url,
                source_type=p_item.source_type, query_used=p_item.query_used, title=p_item.title,
                cleaned_text_content=p_item.main_text_content,
                cleaned_structured_blocks=cleaned_structured_blocks_for_norm,
                language_of_main_text=p_item.detected_language_of_main_text
            )
            normalized_items_all.append(norm_item)
    logger.info(f"Normalized to {len(normalized_items_all)} unique items.")

    _progress_update("Enriching Metadata")
    enriched_items_all: List[EnrichedItem] = []
    gui_enhanced_data_accumulator_for_logger: List[Dict[str, Any]] = []

    for n_item in normalized_items_all:
        item_lang_primary = n_item.language_of_main_text or initial_content_type_hint or 'en'

        enrichment_payload = {
            'categories': list(set([n_item.source_type, (n_item.title or "general")[:20].lower()])),
            'tags': [], 'overall_entities': [],
            'language_of_primary_text': item_lang_primary,
            'quality_score': 7.5, 'complexity_score': 5.0
        }

        primary_text_for_nlp = n_item.cleaned_text_content if n_item.cleaned_text_content else ""

        if primary_text_for_nlp and NLP_MODEL and NLP_LIBS_AVAILABLE:
            try:
                max_len_spacy = NLP_MODEL.max_length - 100
                doc = NLP_MODEL(primary_text_for_nlp[:max_len_spacy])

                enrichment_payload['overall_entities'] = list(
                    set([{'text': ent.text, 'label': ent.label_} for ent in doc.ents]))
                enrichment_payload['tags'] = list(set([token.lemma_.lower() for token in doc if
                                                       not token.is_stop and not token.is_punct and len(
                                                           token.lemma_) > 2][:15]))

                if enrichment_payload['language_of_primary_text'] in ['unknown', 'text', None,
                                                                      'en'] and primary_text_for_nlp.strip():
                    try:
                        detected_lang = detect_language(primary_text_for_nlp[:1500])
                        enrichment_payload['language_of_primary_text'] = detected_lang
                    except LangDetectException:
                        logger.debug(
                            f"Langdetect failed for main text of {n_item.source_url}, keeping '{item_lang_primary}'.")
                        enrichment_payload['language_of_primary_text'] = item_lang_primary
            except Exception as e_spacy:
                logger.warning(f"spaCy/NLP processing failed for main text of {n_item.source_url}: {e_spacy}")

        current_enriched_structured_elements = []
        for block in n_item.cleaned_structured_blocks:
            enriched_block = block.copy()
            if 'language' not in enriched_block and block.get('type') == 'formatted_text_block':
                enriched_block['language'] = block.get('language', 'plaintext')
            current_enriched_structured_elements.append(enriched_block)

        display_meta_summary = {
            'url': str(n_item.source_url), 'title': n_item.title or "Untitled",
            'type': n_item.source_type, 'lang': enrichment_payload['language_of_primary_text'],
            'tags_sample': enrichment_payload['tags'][:5],
            'entities_count': len(enrichment_payload['overall_entities']),
            'structured_blocks_count': len(current_enriched_structured_elements)
        }
        gui_enhanced_data_accumulator_for_logger.append(display_meta_summary)

        enriched_item = EnrichedItem(
            id=n_item.id, normalized_item_id=n_item.id, source_url=n_item.source_url,
            source_type=n_item.source_type, query_used=n_item.query_used, title=n_item.title or "Untitled Content",
            primary_text_content=n_item.cleaned_text_content,
            enriched_structured_elements=current_enriched_structured_elements,
            categories=enrichment_payload['categories'], tags=enrichment_payload['tags'],
            overall_entities=enrichment_payload['overall_entities'],
            language_of_primary_text=enrichment_payload['language_of_primary_text'],
            quality_score=enrichment_payload['quality_score'],
            complexity_score=enrichment_payload['complexity_score'],
            displayable_metadata_summary=display_meta_summary
        )
        enriched_items_all.append(enriched_item)

    logger.info(f"Enriched {len(enriched_items_all)} items.")
    if hasattr(logger, 'enhanced_snippet_data'):
        logger.enhanced_snippet_data = gui_enhanced_data_accumulator_for_logger

    _progress_update("Chunking & Formatting for RAG")
    all_rag_output_items_for_domain: List[RAGOutputItem] = []
    for e_item_to_chunk in enriched_items_all:
        chunks = chunker.chunk_item(e_item_to_chunk)
        all_rag_output_items_for_domain.extend(chunks)
    logger.info(f"Generated {len(all_rag_output_items_for_domain)} RAG-ready items (chunks).")

    _progress_update("Exporting RAG Chunks")
    if all_rag_output_items_for_domain:
        exporter.export_batch(all_rag_output_items_for_domain, export_cfg=current_run_export_config)
    else:
        logger.info("No RAG items to export.")

    fetcher_pool.shutdown()
    logger.info(f"Pipeline finished for: '{domain_query_for_log}'")

    return enriched_items_all


def search_and_fetch(
        query_or_config_path: str,
        logger: logging.Logger,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        content_type_gui: Optional[str] = None
) -> List[EnrichedItem]:
    logger.info(f"search_and_fetch received: '{query_or_config_path}', GUI content_type hint: {content_type_gui}")

    complete_enriched_items = run_pipeline(
        query_or_config_path,
        logger_instance=logger,
        progress_callback=progress_callback,
        initial_content_type_hint=content_type_gui
    )

    if progress_callback: progress_callback("Search and processing complete!", 100)

    return complete_enriched_items


def fetch_stdlib_docs(m, l): logger.warning("Legacy fetch_stdlib_docs called, returns empty."); return []


def fetch_stackoverflow_snippets(q, l, t=None): logger.warning(
    "Legacy fetch_stackoverflow_snippets called, returns empty."); return [], []


def fetch_github_readme_snippets(q, l, mr=None, spr=None): logger.warning(
    "Legacy fetch_github_readme_snippets called, returns empty."); return []


def fetch_github_file_snippets(q, l, mr=None, fprt=None): logger.warning(
    "Legacy fetch_github_file_snippets called, returns empty."); return [], []


def detect_content_type(q: str, l) -> str:
    logger.debug(f"Legacy detect_content_type called for query: {q}")
    if any(ext in q.lower() for ext in ['.pdf']): return 'pdf'
    if any(ext in q.lower() for ext in ['.html', '.htm']): return 'html'
    if any(ext in q.lower() for ext in ['.xml']): return 'xml'
    if any(ext in q.lower() for ext in ['.json']): return 'json'
    if any(ext in q.lower() for ext in ['.txt', '.md']): return 'text'
    return config.DEFAULT_CONTENT_TYPE_FOR_GUI