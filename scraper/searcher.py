# scraper/searcher.py
import os
import logging
import hashlib
import time
import threading
from typing import List, Dict, Optional, Any, Callable, Tuple
import json
import yaml
import re
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
import traceback

import config
from .rag_models import FetchedItem, ParsedItem, NormalizedItem, EnrichedItem, RAGOutputItem, ExtractedLinkInfo
from .fetcher_pool import FetcherPool
from .content_router import ContentRouter
from .chunker import Chunker
from .config_manager import ConfigManager, ExportConfig, SourceConfig

# External dependencies with professional error handling
try:
    from duckduckgo_search import DDGS

    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False
    # Ensure logger is available for this warning, might need basic config if this runs before setup_logger
    logging.getLogger(getattr(config, 'DEFAULT_LOGGER_NAME', 'ModularRAGScraper')).warning(
        "DuckDuckGo search library not available - autonomous search disabled")

try:
    import spacy
    from langdetect import detect as detect_language, LangDetectException

    NLP_LIBS_AVAILABLE = True
except ImportError:
    NLP_LIBS_AVAILABLE = False
    logging.getLogger(getattr(config, 'DEFAULT_LOGGER_NAME', 'ModularRAGScraper')).warning(
        "NLP libraries not available - advanced analysis disabled")


# Simple fallback deduplicator - no external dependencies
class SmartDeduplicator:
    def __init__(self, logger=None):
        self.seen_hashes = set()
        self.logger = logger or logging.getLogger("FallbackDeduplicator")

    def add_snippet(self, text_content, metadata=None):
        h = hashlib.md5(text_content.encode('utf-8', 'replace')).hexdigest()
        return not (h in self.seen_hashes or self.seen_hashes.add(h))

    def is_duplicate(self, text_content):
        h = hashlib.md5(text_content.encode('utf-8', 'replace')).hexdigest()
        return h in self.seen_hashes, "exact_hash" if h in self.seen_hashes else "unique"


# Professional NLP model loading with fallbacks
NLP_MODEL = None
if NLP_LIBS_AVAILABLE:
    try:
        model_name = getattr(config, 'SPACY_MODEL_NAME', 'en_core_web_sm')
        NLP_MODEL = spacy.load(model_name)
        logging.getLogger(getattr(config, 'DEFAULT_LOGGER_NAME', 'ModularRAGScraper')).info(
            f"Loaded spaCy model: {model_name}")
    except OSError:
        logging.getLogger(getattr(config, 'DEFAULT_LOGGER_NAME', 'ModularRAGScraper')).warning(
            f"spaCy model '{model_name}' not found - continuing without advanced NLP")
        NLP_LIBS_AVAILABLE = False
    except Exception as e:
        logging.getLogger(getattr(config, 'DEFAULT_LOGGER_NAME', 'ModularRAGScraper')).error(
            f"Failed to load spaCy model: {e}")
        NLP_LIBS_AVAILABLE = False


@dataclass
class PipelineMetrics:
    """Professional pipeline performance tracking"""
    start_time: datetime
    end_time: Optional[datetime] = None
    total_urls: int = 0
    successful_fetches: int = 0
    failed_fetches: int = 0
    parsed_items: int = 0
    normalized_items: int = 0
    enriched_items: int = 0
    rag_chunks: int = 0
    duplicates_filtered: int = 0
    quality_filtered: int = 0
    errors: List[str] = None  # Store error messages

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def duration(self) -> timedelta:
        end = self.end_time or datetime.now()  # Use current time if end_time not set
        return end - self.start_time

    @property
    def success_rate(self) -> float:
        if self.total_urls == 0:
            return 0.0
        return (self.successful_fetches / self.total_urls) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            'duration_seconds': self.duration.total_seconds(),
            'total_urls': self.total_urls,
            'success_rate': f"{self.success_rate:.1f}%",
            'successful_fetches': self.successful_fetches,
            'failed_fetches': self.failed_fetches,
            'parsed_items': self.parsed_items,
            'normalized_items': self.normalized_items,
            'enriched_items': self.enriched_items,
            'rag_chunks': self.rag_chunks,
            'duplicates_filtered': self.duplicates_filtered,
            'quality_filtered': self.quality_filtered,
            'error_count': len(self.errors)
        }


class ProfessionalQualityFilter:
    """Advanced content quality assessment and filtering"""

    def __init__(self, logger):
        self.logger = logger
        self.quality_thresholds = {
            'minimum_length': getattr(config, 'QUALITY_MIN_LENGTH', 100),
            'substantial_length': getattr(config, 'QUALITY_SUBSTANTIAL_LENGTH', 500),
            'comprehensive_length': getattr(config, 'QUALITY_COMPREHENSIVE_LENGTH', 2000),
            'minimum_score': getattr(config, 'QUALITY_MIN_SCORE', 3)
        }

    def assess_content_quality(self, item: NormalizedItem) -> Tuple[int, Dict[str, Any]]:
        content = (item.cleaned_text_content or '') + ' ' + (item.title or '')
        content_length = len(content)
        quality_details = {'length_score': 0, 'structure_score': 0, 'data_richness_score': 0, 'authority_score': 0,
                           'penalty_score': 0, 'reasons': []}

        if content_length >= self.quality_thresholds['comprehensive_length']:
            quality_details['length_score'] = 4
        elif content_length >= self.quality_thresholds['substantial_length']:
            quality_details['length_score'] = 3
        elif content_length >= self.quality_thresholds['minimum_length']:
            quality_details['length_score'] = 2
        else:
            quality_details['reasons'].append(f"Content too short ({content_length} chars)")

        if item.cleaned_structured_blocks:
            quality_details['structure_score'] = len(item.cleaned_structured_blocks)
            quality_details['reasons'].append(f"Has {len(item.cleaned_structured_blocks)} structured elements")
        if item.custom_fields:
            populated_fields = sum(1 for v in item.custom_fields.values() if v and str(v).strip())
            quality_details['data_richness_score'] = populated_fields * 2
            if populated_fields > 0: quality_details['reasons'].append(f"Rich data: {populated_fields} custom fields")

        url_str = str(item.source_url).lower()
        if any(domain in url_str for domain in ['gov', 'edu', 'org', 'official', 'wikipedia']):
            quality_details['authority_score'] = 2
            quality_details['reasons'].append("Authoritative domain")

        penalty_indicators = ['error 404', 'page not found', 'access denied', 'cookies required', 'javascript required',
                              'please enable', 'subscribe to continue', 'login required', 'paywall']
        penalty_count = sum(1 for indicator in penalty_indicators if indicator in content.lower())
        quality_details['penalty_score'] = -penalty_count * 2
        if penalty_count > 0: quality_details['reasons'].append(f"Quality penalties: {penalty_count}")

        total_score = sum(quality_details[k] for k in
                          ['length_score', 'structure_score', 'data_richness_score', 'authority_score',
                           'penalty_score'])
        return total_score, quality_details

    def filter_by_quality(self, items: List[NormalizedItem]) -> Tuple[List[NormalizedItem], int]:
        if not getattr(config, 'QUALITY_FILTER_ENABLED', True):
            self.logger.info("Quality filter is disabled. Passing all items.")
            return items, 0

        high_quality_items = []
        filtered_count = 0
        for item in items:
            score, details = self.assess_content_quality(item)
            if score >= self.quality_thresholds['minimum_score']:
                high_quality_items.append(item)
                self.logger.debug(f"Quality PASS: {item.source_url} (score: {score})")
            else:
                filtered_count += 1
                self.logger.debug(
                    f"Quality FILTER: {item.source_url} (score: {score}, reasons: {'; '.join(details['reasons'][:3])})")
        self.logger.info(
            f"Quality filter: {len(high_quality_items)}/{len(items)} items passed (filtered: {filtered_count})")
        return high_quality_items, filtered_count


class ProfessionalContentEnricher:
    def __init__(self, nlp_model, logger):
        self.nlp = nlp_model
        self.logger = logger
        self.value_patterns = {
            'numerical_data': r'\d+(?:\.\d+)?(?:%|\$|€|£|pts?|kg|lbs?|mph|km/h)',
            'dates': r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',
        }

    def enrich_item(self, item: NormalizedItem) -> EnrichedItem:
        try:
            categories = self._generate_smart_categories(item)
            tags, keyphrases, entities = self._nlp_process(item)
            language = self._detect_language(item)
            quality_score = self._calculate_quality_score(item, entities, tags)
            complexity_score = self._calculate_complexity_score(item)

            enriched_elements = []
            for element in item.cleaned_structured_blocks:
                enhanced = element.copy()
                enhanced['enriched_at'] = datetime.now().isoformat()
                if 'language' not in enhanced: enhanced['language'] = language
                enriched_elements.append(enhanced)

            metadata_summary = self._create_metadata_summary(item, language, categories, tags, entities,
                                                             enriched_elements, quality_score)

            return EnrichedItem(id=item.id, normalized_item_id=item.id, source_url=item.source_url,
                                source_type=item.source_type, query_used=item.query_used,
                                title=item.title or "Untitled Content", primary_text_content=item.cleaned_text_content,
                                enriched_structured_elements=enriched_elements,
                                custom_fields=item.custom_fields, categories=categories, tags=tags,
                                keyphrases=keyphrases, overall_entities=entities,
                                language_of_primary_text=language, quality_score=quality_score,
                                complexity_score=complexity_score, displayable_metadata_summary=metadata_summary)
        except Exception as e:
            self.logger.error(f"Enrichment failed for {item.source_url}: {e}", exc_info=True)
            return self._create_fallback_enriched_item(item)

    def _nlp_process(self, item: NormalizedItem) -> Tuple[List[str], List[str], List[Dict[str, str]]]:
        tags, keyphrases, entities = set(), [], []
        content = item.cleaned_text_content or ''
        if not (self.nlp and content and NLP_LIBS_AVAILABLE): return sorted(list(tags))[:15], keyphrases[:10], entities[
                                                                                                               :20]

        try:
            doc = self.nlp(content[:5000])  # Limit NLP processing length
            for token in doc:
                if not token.is_stop and not token.is_punct and len(token.lemma_) > 2 and token.pos_ in ['NOUN',
                                                                                                         'PROPN',
                                                                                                         'ADJ']:
                    tags.add(token.lemma_.lower())
            for chunk in doc.noun_chunks:
                phrase = chunk.text.lower().strip()
                if len(phrase.split()) >= 2 and len(phrase) > 5 and not any(
                    sw in phrase for sw in ['this', 'that']): keyphrases.append(phrase)

            seen_entities = set()
            for ent in doc.ents:
                if len(ent.text.strip()) > 2:
                    key = (ent.text.lower().strip(), ent.label_)
                    if key not in seen_entities:
                        entities.append({'text': ent.text.strip(), 'label': ent.label_,
                                         'description': spacy.explain(ent.label_) or ent.label_})
                        seen_entities.add(key)
            keyphrases = sorted(list(set(keyphrases)), key=len, reverse=True)
        except Exception as e:
            self.logger.debug(f"NLP processing sub-step failed: {e}")
        return sorted(list(tags))[:15], keyphrases[:10], entities[:20]

    def _generate_smart_categories(self, item: NormalizedItem) -> List[str]:
        categories = set([item.source_type])
        url_parts = str(item.source_url).lower().split('/')
        domain_parts = url_parts[2].split('.') if len(url_parts) > 2 else []
        for part in domain_parts + url_parts[3:6]:
            if len(part) > 2 and part not in ['www', 'com', 'org', 'net', 'html', 'php']: categories.add(
                part.replace('-', '_'))
        content_len = len(item.cleaned_text_content or '')
        if content_len > 2000:
            categories.add('long_form')
        elif content_len > 500:
            categories.add('standard_length')
        else:
            categories.add('short_form')
        return sorted(list(categories))[:10]

    def _detect_language(self, item: NormalizedItem) -> str:
        content = item.cleaned_text_content or item.title or ''
        if not content.strip(): return 'unknown'
        try:
            return detect_language(content[:1500])
        except LangDetectException:
            self.logger.debug(
                f"Language detection failed for content snippet from {item.source_url}, defaulting to 'en'.")
            return 'en'

    def _calculate_quality_score(self, item: NormalizedItem, entities: List, tags: List) -> float:
        score = 5.0
        content = item.cleaned_text_content or ''
        if len(content) > 2000:
            score += 2.0
        elif len(content) > 1000:
            score += 1.0
        elif len(content) < 100 and not item.custom_fields:
            score -= 2.0  # Penalize short content more if no custom fields
        if item.cleaned_structured_blocks: score += min(len(item.cleaned_structured_blocks) * 0.4,
                                                        1.5)  # slightly less weight
        if item.custom_fields: score += min(sum(1 for v in item.custom_fields.values() if v and str(v).strip()) * 0.5,
                                            2.0)  # more weight for custom fields
        if entities: score += min(len(entities) * 0.1, 1.0)
        if tags: score += min(len(tags) * 0.05, 0.5)
        return min(max(score, 0.5), 10.0)  # Min score can be lower

    def _calculate_complexity_score(self, item: NormalizedItem) -> float:
        content = item.cleaned_text_content or ''
        if not content: return 1.0
        words = content.split()
        num_words = len(words)
        if num_words < 10: return 1.0  # Avoid division by zero for very short texts
        sentences = len(re.split(r'[.!?]+', content))
        avg_word_len = sum(len(w) for w in words) / num_words
        # Simplified Flesch-Kincaid like heuristic
        complexity = 0.39 * (num_words / max(sentences, 1)) + 11.8 * (
            avg_word_len / num_words if num_words else 1) - 15.59
        return min(max(complexity / 10, 1.0), 10.0)  # Normalize to 1-10 range

    def _create_metadata_summary(self, item, lang, cats, tags, ents, struct_elems, qual):
        return {'url': str(item.source_url), 'title': item.title or "N/A", 'source_type': item.source_type,
                'lang': lang,
                'cats': cats[:3], 'top_tags': tags[:3], 'ents': len(ents), 'struct_elems': len(struct_elems),
                'qual': round(qual, 1),
                'ts': datetime.now().isoformat()}

    def _create_fallback_enriched_item(self, item: NormalizedItem) -> EnrichedItem:
        # Fallback if full enrichment fails
        return EnrichedItem(id=item.id, normalized_item_id=item.id, source_url=item.source_url,
                            source_type=item.source_type, query_used=item.query_used,
                            title=item.title or "Untitled", primary_text_content=item.cleaned_text_content,
                            enriched_structured_elements=item.cleaned_structured_blocks,
                            custom_fields=item.custom_fields, categories=[item.source_type, 'fallback_enrichment'],
                            tags=['enrichment_failed'], keyphrases=[], overall_entities=[],
                            language_of_primary_text='unknown', quality_score=2.0, complexity_score=2.0,
                            displayable_metadata_summary={'url': str(item.source_url), 'title': item.title or "N/A",
                                                          'status': 'enrichment_fallback'})


class RobustExporter:
    def __init__(self, logger, default_output_dir: str = None):
        self.logger = logger
        self.default_output_dir = Path(default_output_dir or getattr(config, "DEFAULT_EXPORT_DIR", "./data_exports"))
        self.default_output_dir.mkdir(parents=True, exist_ok=True)

    def export_batch(self, batch_items: List[RAGOutputItem], export_cfg: Optional[ExportConfig] = None) -> bool:
        if not batch_items: self.logger.info("No items to export"); return True
        try:
            output_path, export_format = self._determine_export_path_and_format(batch_items[0], export_cfg)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            valid_items = self._validate_items(batch_items)
            if not valid_items: self.logger.error("No valid items to export after validation"); return False

            success = self._export_items(valid_items, output_path, export_format)
            if success:
                self.logger.info(f"✅ Exported {len(valid_items)} RAG items to {output_path}"); return True
            else:
                self.logger.error(f"❌ Export failed for {output_path}"); return False
        except Exception as e:
            self.logger.error(f"Export batch failed: {e}", exc_info=True); return False

    def _validate_items(self, items: List[RAGOutputItem]) -> List[RAGOutputItem]:
        if not getattr(config, 'EXPORT_VALIDATION_ENABLED', True): return items
        valid_items = []
        for item in items:
            try:
                if not (item.chunk_text and item.chunk_text.strip() and len(item.chunk_text.strip()) >= getattr(config,
                                                                                                                'MIN_CHUNK_SIZE_EXPORT',
                                                                                                                10)):
                    self.logger.debug(f"Skipping item {item.id} due to short/empty chunk text.")
                    continue
                if not item.source_url or not item.source_type:
                    self.logger.debug(f"Skipping item {item.id}: missing required source_url/source_type.")
                    continue
                valid_items.append(item)
            except Exception as e:
                self.logger.warning(f"Item validation failed for {getattr(item, 'id', 'unknown_item')}: {e}")
        self.logger.info(f"Validated {len(valid_items)}/{len(items)} items for export")
        return valid_items

    def _export_items(self, items: List[RAGOutputItem], output_path: str, format_type: str) -> bool:
        try:
            file_exists = Path(output_path).exists()
            with open(output_path, 'a', encoding='utf-8') as f:
                if format_type == "jsonl":
                    return self._export_jsonl(items, f)
                elif format_type == "markdown":
                    return self._export_markdown(items, f, file_exists)
                else:
                    self.logger.error(f"Unsupported export format: {format_type}"); return False
        except Exception as e:
            self.logger.error(f"Export to {output_path} failed: {e}", exc_info=True); return False

    def _export_jsonl(self, items: List[RAGOutputItem], f) -> bool:
        try:
            for item in items: f.write(item.model_dump_json() + '\n')
            return True
        except Exception as e:
            self.logger.error(f"JSONL export process failed: {e}"); return False

    def _export_markdown(self, items: List[RAGOutputItem], f, file_exists: bool) -> bool:
        try:
            if not file_exists and items:  # Write header only if file is new AND there are items
                first = items[0]
                f.write(
                    f"# RAG Export: {first.query_used}\n\n**Source Type:** {first.source_type}\n**Exported:** {datetime.now().isoformat()}\n\n---\n\n")
            for item in items:
                meta = {'id': item.id, 'url': str(item.source_url), 'idx': item.chunk_index,
                        'type': item.chunk_parent_type, 'lang': item.language, 'custom': item.custom_fields}
                f.write(
                    f"## Chunk {item.chunk_index + 1} (ID: {item.id})\n\n```yaml\n{yaml.dump(meta, sort_keys=False, allow_unicode=True)}```\n\n")
                if item.title: f.write(f"**Title:** {item.title}\n\n")
                f.write(
                    f"```{item.language if item.language and item.language not in ['unknown', 'en'] else ''}\n{item.chunk_text}\n```\n\n---\n\n")
            return True
        except Exception as e:
            self.logger.error(f"Markdown export process failed: {e}"); return False

    def _determine_export_path_and_format(self, first_item: RAGOutputItem, export_cfg: Optional[ExportConfig] = None) -> \
    Tuple[str, str]:
        if export_cfg and export_cfg.output_path and export_cfg.format:
            path_str, format_str = Path(export_cfg.output_path), export_cfg.format.lower()
            # Ensure path is absolute or relative to a defined base if not already.
            # If path is relative and doesn't start with '.', assume it's relative to default_output_dir
            if not path_str.is_absolute() and not str(path_str).startswith(('.', os.sep, os.altsep or '')):
                path_str = self.default_output_dir / path_str
            else:  # If it's absolute or starts with '.', create parent dirs
                path_str.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Using configured export: {str(path_str)} ({format_str})")
            return str(path_str), format_str

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        q_safe = re.sub(r'[^\w\-_.]', '_', first_item.query_used)[:30]
        s_safe = re.sub(r'[^\w\-_.]', '_', first_item.source_type)[:20]
        export_dir = self.default_output_dir / f"{q_safe}_{s_safe}"
        export_dir.mkdir(parents=True, exist_ok=True)
        path_str = str(export_dir / f"rag_export_{ts}.jsonl")
        self.logger.info(f"Using default export path: {path_str} (jsonl)")
        return path_str, "jsonl"


def _clean_text_for_dedup(text: Optional[str]) -> str:
    if not text: return ""
    return re.sub(r'\s+', ' ', text.lower().strip())


def run_professional_pipeline(
        query_or_config_path: str,
        logger_instance=None,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        initial_content_type_hint: Optional[str] = None,
        max_retries: int = 3  # Not used yet, but good for future
) -> Tuple[List[EnrichedItem], PipelineMetrics]:
    logger = logger_instance if logger_instance else logging.getLogger(
        getattr(config, 'DEFAULT_LOGGER_NAME', 'ModularRAGScraper'))
    if not logger.handlers:  # Basic setup if no handlers are configured (e.g. running standalone)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    metrics = PipelineMetrics(start_time=datetime.now())
    fetcher_pool = None  # Define fetcher_pool here for finally block

    def update_progress(step_name: str, current_step: int, total_steps: int = 10):
        percentage = int((current_step / total_steps) * 100)
        if progress_callback:
            progress_callback(f"{step_name} ({current_step}/{total_steps})", percentage)
        logger.info(f"🔄 Pipeline progress: {step_name} - {percentage}%")

    try:
        update_progress("Initializing Configuration", 1)

        logger.info(f"DEBUG: Received input for pipeline: '{query_or_config_path}'")
        path_exists = os.path.exists(query_or_config_path)
        logger.debug(f"DEBUG: Path '{query_or_config_path}' exists? {path_exists}")
        is_valid_extension = query_or_config_path.lower().endswith((".yaml", ".yml", ".json"))
        logger.debug(f"DEBUG: Path '{query_or_config_path}' has valid config extension? {is_valid_extension}")

        is_config_file_mode = path_exists and is_valid_extension
        logger.info(f"DEBUG: Determined is_config_file_mode: {is_config_file_mode}")

        cfg_manager: ConfigManager
        domain_query_for_log = query_or_config_path

        if is_config_file_mode:
            logger.info(f"Attempting to initialize ConfigManager with config file: '{query_or_config_path}'")
            cfg_manager = ConfigManager(config_path=query_or_config_path, logger_instance=logger)
            if not cfg_manager.config:  # This checks if self.config was set in ConfigManager's __init__
                error_msg = f"Failed to load or validate configuration file: '{query_or_config_path}'"
                metrics.errors.append(error_msg)
                logger.error(f"❌ {error_msg}. Check ConfigManager logs for Pydantic validation errors.")
                return [], metrics  # Exit early if config loading fails
            logger.info(f"Successfully loaded and validated config: '{query_or_config_path}'")
            if cfg_manager.config.domain_info and cfg_manager.config.domain_info.get('name'):
                domain_query_for_log = cfg_manager.config.domain_info.get('name')
        else:
            logger.info(
                f"Input '{query_or_config_path}' is not a config file. Initializing ConfigManager without a path (for potential programmatic config or defaults).")
            cfg_manager = ConfigManager(
                logger_instance=logger)  # No path, will use defaults or expect programmatic config

        logger.info(f"🚀 Professional pipeline starting for: '{domain_query_for_log}'")

        update_progress("Initializing Components", 2)
        try:
            fetcher_pool = FetcherPool(num_workers=getattr(config, 'MAX_CONCURRENT_FETCHERS', 3), logger=logger)
            content_router = ContentRouter(config_manager=cfg_manager,
                                           logger_instance=logger)  # Pass the (potentially loaded) cfg_manager
            deduplicator = SmartDeduplicator(logger=logger)
            chunker = Chunker(logger_instance=logger)
            exporter = RobustExporter(logger=logger)
            quality_filter = ProfessionalQualityFilter(logger=logger)
            enricher = ProfessionalContentEnricher(nlp_model=NLP_MODEL, logger=logger)
        except Exception as e:
            metrics.errors.append(f"Component initialization failed: {e}")
            logger.error(f"❌ Component initialization failed: {e}", exc_info=True)
            return [], metrics

        update_progress("Preparing Fetch Tasks", 3)
        current_run_export_config: Optional[ExportConfig] = None
        tasks_to_fetch = []

        if is_config_file_mode and cfg_manager.config and cfg_manager.config.sources:
            logger.info(f"CONFIG MODE: Preparing tasks from {len(cfg_manager.config.sources)} configured sources.")
            for src_cfg_model in cfg_manager.get_sources():  # Safe now due to check above
                if not current_run_export_config and src_cfg_model.export_config:
                    current_run_export_config = src_cfg_model.export_config
                logger.debug(f"Preparing tasks for source: '{src_cfg_model.name}'. Seeds: {len(src_cfg_model.seeds)}")
                for seed_url_model in src_cfg_model.seeds:
                    tasks_to_fetch.append(
                        (str(seed_url_model), src_cfg_model.source_type or src_cfg_model.name, domain_query_for_log,
                         None)
                    )
        elif not is_config_file_mode:
            logger.info(f"QUERY/URL MODE: Processing input '{query_or_config_path}'.")
            query_str = query_or_config_path
            query_sanitized = re.sub(r'[^\w\-_.]', '_', query_str)[:50]  # Sanitize for dir name
            export_dir = Path(getattr(config, "DEFAULT_EXPORT_DIR", "./data_exports")) / f"query_{query_sanitized}"
            export_dir.mkdir(parents=True, exist_ok=True)
            current_run_export_config = ExportConfig(output_path=str(export_dir / "rag_export.jsonl"), format="jsonl")

            if query_str.startswith(("http://", "https://")):
                tasks_to_fetch.append((query_str, "direct_url_query", query_str, None))
            elif DUCKDUCKGO_AVAILABLE:
                logger.info(f"🔍 Performing autonomous search for: '{query_str}'")
                try:
                    search_delay = getattr(config, 'DUCKDUCKGO_SEARCH_DELAY', 1.5)
                    logger.debug(f"Waiting {search_delay}s before DuckDuckGo search.")
                    time.sleep(search_delay)
                    with DDGS(timeout=10) as ddgs:
                        ddgs_results = list(
                            ddgs.text(query_str, max_results=getattr(config, 'AUTONOMOUS_SEARCH_MAX_RESULTS', 5)))
                    for res in ddgs_results:
                        if res.get('href'): tasks_to_fetch.append(
                            (res['href'], "autonomous_web_search", query_str, res.get('title')))
                except Exception as e_search:
                    error_msg = f"Autonomous search failed for query '{query_str}': {e_search}"
                    metrics.errors.append(error_msg)
                    logger.error(f"❌ {error_msg}", exc_info=True)
                    return [], metrics
            else:  # Not a URL, and DDG not available
                error_msg = f"Input '{query_str}' is not a URL, and autonomous search (DuckDuckGo) is not available."
                metrics.errors.append(error_msg)
                logger.error(f"❌ {error_msg}")
                return [], metrics
        else:  # This should mean is_config_file_mode was true, but cfg_manager.config was None (handled above) or no sources
            error_msg = "Config file mode determined, but no sources found in the configuration or config failed to load properly."
            metrics.errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
            return [], metrics

        if not tasks_to_fetch:
            metrics.errors.append("No URLs were prepared for fetching.")
            logger.error("❌ No URLs prepared for fetching. Check config or query.")
            return [], metrics

        metrics.total_urls = len(tasks_to_fetch)
        logger.info(f"📋 Prepared {metrics.total_urls} URLs for fetching.")
        for url, source_type, query_used_log, item_title in tasks_to_fetch:
            fetcher_pool.submit_task(url, source_type, query_used_log, item_title)

        update_progress(f"Fetching Content ({metrics.total_urls} URLs)", 4)
        fetched_items_all: List[FetchedItem] = fetcher_pool.get_results()
        metrics.successful_fetches = len(fetched_items_all)
        metrics.failed_fetches = metrics.total_urls - metrics.successful_fetches
        if not fetched_items_all:
            metrics.errors.append("No content was successfully fetched from any URL.")
            logger.warning("⚠️ No content was successfully fetched.")
            # No need to return here, subsequent steps will handle empty lists
        logger.info(f"✅ Fetched {metrics.successful_fetches} items (success rate: {metrics.success_rate:.1f}%)")

        update_progress("Parsing Content", 5)
        parsed_items_all: List[ParsedItem] = []
        for item_fetched in fetched_items_all:
            if item_fetched.content_bytes or item_fetched.content:  # Ensure there's something to parse
                try:
                    parsed = content_router.route_and_parse(item_fetched)
                    if parsed: parsed_items_all.append(parsed)
                except Exception as e_parse:
                    metrics.errors.append(f"Parse error for {item_fetched.source_url}: {e_parse}")
                    logger.warning(f"⚠️ Parse failed for {item_fetched.source_url}: {e_parse}", exc_info=True)
        metrics.parsed_items = len(parsed_items_all)
        logger.info(f"✅ Parsed {metrics.parsed_items} items.")

        update_progress("Normalizing & Deduplicating", 6)
        normalized_items_all: List[NormalizedItem] = []
        for p_item in parsed_items_all:
            try:
                text_for_dedup_parts = []
                if p_item.main_text_content: text_for_dedup_parts.append(
                    _clean_text_for_dedup(p_item.main_text_content))
                for block in p_item.extracted_structured_blocks:  # Consider structured content for dedup
                    block_content = block.get('content', '')
                    if block.get('type') == 'semantic_figure_with_caption':  # Handle special structure
                        block_content = f"{block.get('figure_content', '')} {block.get('caption_content', '')}".strip()
                    if block_content: text_for_dedup_parts.append(_clean_text_for_dedup(str(block_content)))

                full_content_signature = " ".join(filter(None, text_for_dedup_parts)).strip()

                is_dup = False
                if full_content_signature:
                    is_dup, _ = deduplicator.is_duplicate(full_content_signature)
                elif not p_item.custom_fields and not p_item.extracted_structured_blocks:  # If no text and no custom/structured, might be a dup if seen before
                    is_dup, _ = deduplicator.is_duplicate(f"empty_content_placeholder_for_{p_item.source_url}")

                if not is_dup:
                    if full_content_signature:
                        deduplicator.add_snippet(full_content_signature)
                    elif not p_item.custom_fields and not p_item.extracted_structured_blocks:
                        deduplicator.add_snippet(f"empty_content_placeholder_for_{p_item.source_url}")

                    norm_item = NormalizedItem(
                        id=p_item.id, parsed_item_id=p_item.id, source_url=p_item.source_url,
                        source_type=p_item.source_type,
                        query_used=p_item.query_used, title=p_item.title, cleaned_text_content=p_item.main_text_content,
                        cleaned_structured_blocks=p_item.extracted_structured_blocks,
                        custom_fields=p_item.custom_fields,
                        language_of_main_text=p_item.detected_language_of_main_text  # Propagate from parser
                    )
                    normalized_items_all.append(norm_item)
                else:
                    metrics.duplicates_filtered += 1
                    logger.debug(f"Filtered duplicate for {p_item.source_url}")
            except Exception as e_norm:
                metrics.errors.append(f"Normalization/Deduplication error for {p_item.source_url}: {e_norm}")
                logger.warning(f"⚠️ Normalization/Deduplication failed for {p_item.source_url}: {e_norm}",
                               exc_info=True)
        metrics.normalized_items = len(normalized_items_all)
        logger.info(
            f"✅ Normalized {metrics.normalized_items} unique items (filtered {metrics.duplicates_filtered} duplicates)")

        update_progress("Quality Filtering", 7)
        high_quality_items, filtered_count = quality_filter.filter_by_quality(normalized_items_all)
        metrics.quality_filtered = filtered_count

        update_progress("Enriching Metadata", 8)
        enriched_items_all: List[EnrichedItem] = []
        for n_item in high_quality_items:  # Iterate over quality-filtered items
            try:
                enriched_items_all.append(enricher.enrich_item(n_item))
            except Exception as e_enrich:
                metrics.errors.append(f"Enrichment error for {n_item.source_url}: {e_enrich}")
                logger.warning(f"⚠️ Enrichment failed for {n_item.source_url}: {e_enrich}", exc_info=True)
                enriched_items_all.append(enricher._create_fallback_enriched_item(n_item))  # Add fallback
        metrics.enriched_items = len(enriched_items_all)
        logger.info(f"✅ Enriched {metrics.enriched_items} items")
        if hasattr(logger, 'enhanced_snippet_data'):  # For GUI
            logger.enhanced_snippet_data = [item.displayable_metadata_summary for item in enriched_items_all]

        update_progress("Chunking & Formatting for RAG", 9)
        all_rag_chunks: List[RAGOutputItem] = []
        for e_item in enriched_items_all:
            try:
                all_rag_chunks.extend(chunker.chunk_item(e_item))
            except Exception as e_chunk:
                metrics.errors.append(f"Chunking error for {e_item.source_url}: {e_chunk}")
                logger.warning(f"⚠️ Chunking failed for {e_item.source_url}: {e_chunk}", exc_info=True)
        metrics.rag_chunks = len(all_rag_chunks)
        logger.info(f"✅ Generated {metrics.rag_chunks} RAG chunks")

        update_progress("Exporting RAG Data", 10)
        if all_rag_chunks:
            if not exporter.export_batch(all_rag_chunks, export_cfg=current_run_export_config):
                metrics.errors.append("Export process reported failure.")  # Exporter logs specifics
        else:
            logger.info("ℹ️ No RAG chunks to export.")

        metrics.end_time = datetime.now()
        logger.info(f"""
🎉 Professional pipeline completed successfully!
📊 Summary:
   • Duration: {metrics.duration.total_seconds():.1f}s
   • URLs Attempted: {metrics.total_urls}
   • Successful Fetches: {metrics.successful_fetches} (Rate: {metrics.success_rate:.1f}%)
   • Parsed Items: {metrics.parsed_items}
   • Unique Items (post-dedup): {metrics.normalized_items}
   • Quality Filtered Out: {metrics.quality_filtered}
   • Enriched Items: {metrics.enriched_items}
   • RAG Chunks Generated: {metrics.rag_chunks}
   • Errors during pipeline: {len(metrics.errors)}
""")
        if metrics.errors:
            logger.warning(f"⚠️ {len(metrics.errors)} errors logged during processing. First few:")
            for i, err_msg in enumerate(metrics.errors[:min(5, len(metrics.errors))], 1): logger.warning(
                f"  {i}. {err_msg}")

        return enriched_items_all, metrics

    except KeyboardInterrupt:  # Graceful exit on Ctrl+C
        error_msg = "Pipeline interrupted by user (KeyboardInterrupt)."
        logger.warning(f"🛑 {error_msg}")
        if metrics: metrics.errors.append(error_msg); metrics.end_time = datetime.now()
        return [], metrics
    except Exception as e_pipeline:  # Catch-all for unexpected pipeline errors
        error_msg = f"Critical pipeline failure: {e_pipeline}"
        logger.error(f"💥 {error_msg}", exc_info=True)
        if metrics: metrics.errors.append(error_msg); metrics.end_time = datetime.now()
        return [], metrics
    finally:  # Ensure fetcher_pool is shut down
        if fetcher_pool:
            logger.info("Ensuring FetcherPool executor is shut down in finally block.")
            fetcher_pool.shutdown()


def search_and_fetch(
        query_or_config_path: str,
        logger: logging.Logger,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        content_type_gui: Optional[str] = None  # Hint from GUI
) -> List[EnrichedItem]:
    """
    Main entry point for the scraping pipeline.
    Determines if input is a config file path or a query/URL.
    Returns a list of EnrichedItem objects for potential GUI display or further use.
    """
    if not query_or_config_path or not query_or_config_path.strip():
        logger.error("❌ No query or config path provided to search_and_fetch.")
        if progress_callback: progress_callback("Error: No input", 100)
        return []

    logger.info(f"🎯 Professional search initiated with input: '{query_or_config_path}' (GUI hint: {content_type_gui})")

    enriched_items, metrics_data = run_professional_pipeline(
        query_or_config_path,
        logger_instance=logger,
        progress_callback=progress_callback,
        initial_content_type_hint=content_type_gui  # Pass GUI hint
    )

    # Log final metrics from the pipeline run
    if metrics_data:  # metrics_data should always be returned
        metrics_summary = metrics_data.to_dict()
        logger.info(f"📈 Final Pipeline Metrics: {json.dumps(metrics_summary, indent=2, ensure_ascii=False)}")
    else:  # Should not happen if run_professional_pipeline is robust
        logger.error("❌ Pipeline did not return metrics data.")

    if progress_callback:  # Ensure progress shows 100% at the very end
        final_msg = "Processing complete!"
        if metrics_data and metrics_data.errors:
            final_msg = f"Processing complete with {len(metrics_data.errors)} error(s). Check logs."
        elif not enriched_items and not (
                metrics_data and metrics_data.errors):  # No items, no errors - likely no URLs or all filtered
            final_msg = "Processing complete. No data extracted or all items filtered. Check logs/config."
        progress_callback(final_msg, 100)

    return enriched_items


# Legacy compatibility functions (can be removed if not used elsewhere)
def fetch_stdlib_docs(m, l): return []


def fetch_stackoverflow_snippets(q, l, t=None): return [], []


def fetch_github_readme_snippets(q, l, mr=None, spr=None): return []


def fetch_github_file_snippets(q, l, mr=None, fprt=None): return [], []


def detect_content_type(q: str, l): return getattr(config, 'DEFAULT_CONTENT_TYPE_FOR_GUI', 'html')