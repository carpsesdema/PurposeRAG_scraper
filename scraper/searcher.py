# scraper/searcher.py - QUICK FIX VERSION
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
    logging.getLogger(config.DEFAULT_LOGGER_NAME).warning(
        "DuckDuckGo search library not available - autonomous search disabled")

try:
    import spacy
    from langdetect import detect as detect_language, LangDetectException
    NLP_LIBS_AVAILABLE = True
except ImportError:
    NLP_LIBS_AVAILABLE = False
    logging.getLogger(config.DEFAULT_LOGGER_NAME).warning("NLP libraries not available - advanced analysis disabled")

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
        logging.getLogger(config.DEFAULT_LOGGER_NAME).info(f"Loaded spaCy model: {model_name}")
    except OSError:
        logging.getLogger(config.DEFAULT_LOGGER_NAME).warning(
            f"spaCy model '{model_name}' not found - continuing without advanced NLP")
        NLP_LIBS_AVAILABLE = False
    except Exception as e:
        logging.getLogger(config.DEFAULT_LOGGER_NAME).error(f"Failed to load spaCy model: {e}")
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
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def duration(self) -> timedelta:
        end = self.end_time or datetime.now()
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
            'minimum_length': 100,
            'substantial_length': 500,
            'comprehensive_length': 2000,
            'minimum_score': 3
        }

    def assess_content_quality(self, item: NormalizedItem) -> Tuple[int, Dict[str, Any]]:
        """
        Professional content quality assessment
        Returns: (quality_score, quality_details)
        """
        content = (item.cleaned_text_content or '') + ' ' + (item.title or '')
        content_length = len(content)

        quality_details = {
            'length_score': 0,
            'structure_score': 0,
            'data_richness_score': 0,
            'authority_score': 0,
            'penalty_score': 0,
            'reasons': []
        }

        # Length-based scoring
        if content_length >= self.quality_thresholds['comprehensive_length']:
            quality_details['length_score'] = 4
        elif content_length >= self.quality_thresholds['substantial_length']:
            quality_details['length_score'] = 3
        elif content_length >= self.quality_thresholds['minimum_length']:
            quality_details['length_score'] = 2
        else:
            quality_details['length_score'] = 0
            quality_details['reasons'].append(f"Content too short ({content_length} chars)")

        # Structural richness
        if item.cleaned_structured_blocks:
            quality_details['structure_score'] = len(item.cleaned_structured_blocks)
            quality_details['reasons'].append(f"Has {len(item.cleaned_structured_blocks)} structured elements")

        # Data richness (custom fields extracted)
        if item.custom_fields:
            populated_fields = sum(1 for v in item.custom_fields.values() if v and str(v).strip())
            quality_details['data_richness_score'] = populated_fields * 2
            if populated_fields > 0:
                quality_details['reasons'].append(f"Rich data: {populated_fields} custom fields")

        # Authority indicators
        url_str = str(item.source_url).lower()
        authority_domains = ['gov', 'edu', 'org', 'official', 'wikipedia']
        if any(domain in url_str for domain in authority_domains):
            quality_details['authority_score'] = 2
            quality_details['reasons'].append("Authoritative domain")

        # Penalty factors
        content_lower = content.lower()
        penalty_indicators = [
            'error 404', 'page not found', 'access denied', 'cookies required',
            'javascript required', 'please enable', 'subscribe to continue',
            'login required', 'paywall', 'premium content'
        ]

        penalty_count = sum(1 for indicator in penalty_indicators if indicator in content_lower)
        quality_details['penalty_score'] = -penalty_count * 2

        if penalty_count > 0:
            quality_details['reasons'].append(f"Quality penalties: {penalty_count}")

        # Calculate total score
        total_score = (
                quality_details['length_score'] +
                quality_details['structure_score'] +
                quality_details['data_richness_score'] +
                quality_details['authority_score'] +
                quality_details['penalty_score']
        )

        return total_score, quality_details

    def filter_by_quality(self, items: List[NormalizedItem]) -> Tuple[List[NormalizedItem], int]:
        """Filter items by quality, return (filtered_items, filtered_count)"""
        high_quality_items = []
        filtered_count = 0

        for item in items:
            score, details = self.assess_content_quality(item)

            if score >= self.quality_thresholds['minimum_score']:
                high_quality_items.append(item)
                self.logger.debug(f"Quality PASS: {item.source_url} (score: {score})")
            else:
                filtered_count += 1
                reasons = '; '.join(details['reasons'][:3])
                self.logger.debug(f"Quality FILTER: {item.source_url} (score: {score}, reasons: {reasons})")

        self.logger.info(
            f"Quality filter: {len(high_quality_items)}/{len(items)} items passed (filtered: {filtered_count})")
        return high_quality_items, filtered_count

class ProfessionalContentEnricher:
    """Enhanced content enrichment with domain intelligence"""

    def __init__(self, nlp_model, logger):
        self.nlp = nlp_model
        self.logger = logger

        # Domain-agnostic value indicators
        self.value_patterns = {
            'numerical_data': r'\d+(?:\.\d+)?(?:%|\$|€|£|pts?|kg|lbs?|mph|km/h)',
            'dates': r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',
            'names': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',
            'locations': r'\b[A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*\b',
            'organizations': r'\b[A-Z]{2,}(?:\s+[A-Z][a-z]+)*\b'
        }

    def enrich_item(self, item: NormalizedItem) -> EnrichedItem:
        """Professional content enrichment with error handling"""
        try:
            # Initialize enrichment data
            enrichment_data = {
                'categories': self._generate_smart_categories(item),
                'tags': self._extract_smart_tags(item),
                'keyphrases': self._extract_keyphrases(item),
                'entities': self._extract_entities(item),
                'language': self._detect_language(item),
                'quality_score': self._calculate_quality_score(item),
                'complexity_score': self._calculate_complexity_score(item)
            }

            # Create enriched structured elements
            enriched_elements = []
            for element in item.cleaned_structured_blocks:
                enhanced_element = element.copy()
                enhanced_element['enriched_at'] = datetime.now().isoformat()
                if 'language' not in enhanced_element:
                    enhanced_element['language'] = enrichment_data['language']
                enriched_elements.append(enhanced_element)

            # Generate metadata summary for display
            metadata_summary = {
                'url': str(item.source_url),
                'title': item.title or "Untitled",
                'source_type': item.source_type,
                'language': enrichment_data['language'],
                'categories': enrichment_data['categories'][:3],
                'top_tags': enrichment_data['tags'][:5],
                'top_keyphrases': enrichment_data['keyphrases'][:3],
                'entity_count': len(enrichment_data['entities']),
                'custom_fields_count': len([v for v in item.custom_fields.values() if v]),
                'structured_elements': len(enriched_elements),
                'quality_score': enrichment_data['quality_score'],
                'processing_timestamp': datetime.now().isoformat()
            }

            return EnrichedItem(
                id=item.id,
                normalized_item_id=item.id,
                source_url=item.source_url,
                source_type=item.source_type,
                query_used=item.query_used,
                title=item.title or "Untitled Content",
                primary_text_content=item.cleaned_text_content,
                enriched_structured_elements=enriched_elements,
                custom_fields=item.custom_fields,
                categories=enrichment_data['categories'],
                tags=enrichment_data['tags'],
                keyphrases=enrichment_data['keyphrases'],
                overall_entities=enrichment_data['entities'],
                language_of_primary_text=enrichment_data['language'],
                quality_score=enrichment_data['quality_score'],
                complexity_score=enrichment_data['complexity_score'],
                displayable_metadata_summary=metadata_summary
            )

        except Exception as e:
            self.logger.error(f"Enrichment failed for {item.source_url}: {e}", exc_info=True)
            # Return basic enriched item on failure
            return self._create_fallback_enriched_item(item)

    def _generate_smart_categories(self, item: NormalizedItem) -> List[str]:
        """Automatically generate relevant categories"""
        categories = set()

        # Source-based categorization
        categories.add(item.source_type)

        # URL-based categorization
        url_parts = str(item.source_url).lower().split('/')
        domain_parts = url_parts[2].split('.') if len(url_parts) > 2 else []

        for part in domain_parts + url_parts[3:6]:
            if len(part) > 2 and part not in ['www', 'com', 'org', 'net']:
                categories.add(part.replace('-', '_'))

        # Content-based categorization
        content = (item.cleaned_text_content or '').lower()
        if len(content) > 500:
            categories.add('comprehensive')
        elif len(content) > 100:
            categories.add('standard')
        else:
            categories.add('brief')

        return sorted(list(categories))[:10]

    def _extract_smart_tags(self, item: NormalizedItem) -> List[str]:
        """Extract meaningful tags using NLP and patterns"""
        tags = set()
        content = item.cleaned_text_content or ''

        # Pattern-based extraction
        for pattern_name, pattern in self.value_patterns.items():
            if re.search(pattern, content):
                tags.add(pattern_name)

        # NLP-based extraction only if available
        if self.nlp and content and NLP_LIBS_AVAILABLE:
            try:
                doc = self.nlp(content[:2000])

                # Extract significant terms
                for token in doc:
                    if (not token.is_stop and not token.is_punct and
                            len(token.lemma_) > 2 and token.pos_ in ['NOUN', 'ADJ']):
                        tags.add(token.lemma_.lower())

                # Limit tags to most relevant
                if len(tags) > 20:
                    entity_lemmas = {token.lemma_.lower() for ent in doc.ents for token in ent}
                    high_freq_lemmas = {token.lemma_.lower() for token in doc
                                        if doc.text.lower().count(token.lemma_.lower()) > 1}
                    tags = entity_lemmas.union(high_freq_lemmas)

            except Exception as e:
                self.logger.debug(f"NLP tag extraction failed: {e}")

        return sorted(list(tags))[:15]

    def _extract_keyphrases(self, item: NormalizedItem) -> List[str]:
        """Extract key phrases from content"""
        keyphrases = []
        content = item.cleaned_text_content or ''

        if self.nlp and content and NLP_LIBS_AVAILABLE:
            try:
                doc = self.nlp(content[:3000])

                # Extract noun chunks as keyphrases
                for chunk in doc.noun_chunks:
                    phrase = chunk.text.lower().strip()
                    if (len(phrase.split()) >= 2 and len(phrase) > 5 and
                            not any(stop_word in phrase for stop_word in ['this', 'that', 'these', 'those'])):
                        keyphrases.append(phrase)

                # Remove duplicates and sort by length
                keyphrases = sorted(list(set(keyphrases)), key=len, reverse=True)

            except Exception as e:
                self.logger.debug(f"Keyphrase extraction failed: {e}")

        return keyphrases[:10]

    def _extract_entities(self, item: NormalizedItem) -> List[Dict[str, str]]:
        """Extract named entities"""
        entities = []
        content = item.cleaned_text_content or ''

        if self.nlp and content and NLP_LIBS_AVAILABLE:
            try:
                doc = self.nlp(content[:3000])

                for ent in doc.ents:
                    if len(ent.text.strip()) > 2:
                        entities.append({
                            'text': ent.text.strip(),
                            'label': ent.label_,
                            'description': spacy.explain(ent.label_) or ent.label_
                        })

                # Remove duplicates
                seen = set()
                unique_entities = []
                for ent in entities:
                    key = (ent['text'].lower(), ent['label'])
                    if key not in seen:
                        seen.add(key)
                        unique_entities.append(ent)

                entities = unique_entities

            except Exception as e:
                self.logger.debug(f"Entity extraction failed: {e}")

        return entities[:20]

    def _detect_language(self, item: NormalizedItem) -> str:
        """Detect content language with fallbacks"""
        content = item.cleaned_text_content or ''

        if not content.strip():
            return 'unknown'

        try:
            from langdetect import detect
            detected = detect(content[:1500])
            return detected
        except:
            return 'en'  # Default fallback

    def _calculate_quality_score(self, item: NormalizedItem) -> float:
        """Calculate content quality score"""
        score = 5.0  # Base score

        content = item.cleaned_text_content or ''

        # Length bonus
        if len(content) > 2000:
            score += 2.0
        elif len(content) > 1000:
            score += 1.0
        elif len(content) < 100:
            score -= 2.0

        # Structure bonus
        if item.cleaned_structured_blocks:
            score += min(len(item.cleaned_structured_blocks) * 0.5, 2.0)

        # Custom fields bonus
        if item.custom_fields:
            populated = sum(1 for v in item.custom_fields.values() if v and str(v).strip())
            score += min(populated * 0.3, 1.5)

        return min(max(score, 1.0), 10.0)

    def _calculate_complexity_score(self, item: NormalizedItem) -> float:
        """Calculate content complexity score"""
        content = item.cleaned_text_content or ''

        if not content:
            return 1.0

        # Simple complexity metrics
        sentences = len(re.split(r'[.!?]+', content))
        words = len(content.split())
        avg_word_length = sum(len(word) for word in content.split()) / max(words, 1)

        complexity = min((sentences / max(words / 20, 1)) * avg_word_length / 5, 10.0)
        return max(complexity, 1.0)

    def _create_fallback_enriched_item(self, item: NormalizedItem) -> EnrichedItem:
        """Create minimal enriched item when enrichment fails"""
        return EnrichedItem(
            id=item.id,
            normalized_item_id=item.id,
            source_url=item.source_url,
            source_type=item.source_type,
            query_used=item.query_used,
            title=item.title or "Untitled Content",
            primary_text_content=item.cleaned_text_content,
            enriched_structured_elements=item.cleaned_structured_blocks,
            custom_fields=item.custom_fields,
            categories=[item.source_type, 'fallback'],
            tags=['processing_failed'],
            keyphrases=[],
            overall_entities=[],
            language_of_primary_text='unknown',
            quality_score=3.0,
            complexity_score=3.0,
            displayable_metadata_summary={
                'url': str(item.source_url),
                'title': item.title or "Untitled",
                'source_type': item.source_type,
                'processing_status': 'fallback_mode'
            }
        )

class RobustExporter:
    """Professional-grade export system with error handling and validation"""

    def __init__(self, logger, default_output_dir: str = "./data_exports"):
        self.logger = logger
        self.default_output_dir = Path(default_output_dir)
        self.default_output_dir.mkdir(parents=True, exist_ok=True)

    def export_batch(self, batch_items: List[RAGOutputItem], export_cfg: Optional[ExportConfig] = None) -> bool:
        """Export batch with comprehensive error handling"""
        if not batch_items:
            self.logger.info("No items to export")
            return True

        try:
            output_path, export_format = self._determine_export_path_and_format(batch_items[0], export_cfg)

            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Validate items before export
            valid_items = self._validate_items(batch_items)
            if not valid_items:
                self.logger.error("No valid items to export after validation")
                return False

            # Perform export with format-specific handling
            success = self._export_items(valid_items, output_path, export_format)

            if success:
                self.logger.info(f"✅ Exported {len(valid_items)} RAG items to {output_path}")
                return True
            else:
                self.logger.error(f"❌ Export failed for {output_path}")
                return False

        except Exception as e:
            self.logger.error(f"Export batch failed: {e}", exc_info=True)
            return False

    def _validate_items(self, items: List[RAGOutputItem]) -> List[RAGOutputItem]:
        """Validate items before export"""
        valid_items = []

        for item in items:
            try:
                # Basic validation
                if not item.chunk_text or not item.chunk_text.strip():
                    self.logger.debug(f"Skipping item {item.id}: empty chunk text")
                    continue

                if len(item.chunk_text.strip()) < 10:
                    self.logger.debug(f"Skipping item {item.id}: chunk too short")
                    continue

                # Validate required fields
                if not item.source_url or not item.source_type:
                    self.logger.debug(f"Skipping item {item.id}: missing required fields")
                    continue

                valid_items.append(item)

            except Exception as e:
                self.logger.warning(f"Item validation failed for {getattr(item, 'id', 'unknown')}: {e}")
                continue

        self.logger.info(f"Validated {len(valid_items)}/{len(items)} items for export")
        return valid_items

    def _export_items(self, items: List[RAGOutputItem], output_path: str, format_type: str) -> bool:
        """Export items in specified format"""
        try:
            file_exists = Path(output_path).exists()

            with open(output_path, 'a', encoding='utf-8') as f:
                if format_type == "jsonl":
                    return self._export_jsonl(items, f)
                elif format_type == "markdown":
                    return self._export_markdown(items, f, file_exists)
                else:
                    self.logger.error(f"Unsupported export format: {format_type}")
                    return False

        except Exception as e:
            self.logger.error(f"Export to {output_path} failed: {e}", exc_info=True)
            return False

    def _export_jsonl(self, items: List[RAGOutputItem], file_handle) -> bool:
        """Export as JSONL format"""
        try:
            for item in items:
                json_line = item.model_dump_json()
                file_handle.write(json_line + '\n')
            return True
        except Exception as e:
            self.logger.error(f"JSONL export failed: {e}")
            return False

    def _export_markdown(self, items: List[RAGOutputItem], file_handle, file_exists: bool) -> bool:
        """Export as Markdown format"""
        try:
            if not file_exists:
                # Write header for new file
                first_item = items[0]
                header = f"""# RAG Export: {first_item.query_used}

**Source Type:** {first_item.source_type}  
**Exported:** {datetime.now().isoformat()}  
**Total Chunks:** {len(items)}

---

"""
                file_handle.write(header)

            for item in items:
                # Create metadata block
                metadata = {
                    'chunk_id': item.id,
                    'source_url': str(item.source_url),
                    'chunk_index': item.chunk_index,
                    'parent_type': item.chunk_parent_type,
                    'language': item.language,
                    'categories': item.categories,
                    'custom_fields': item.custom_fields
                }

                metadata_yaml = yaml.dump(metadata, default_flow_style=False, allow_unicode=True)

                # Write chunk
                file_handle.write(f"## Chunk {item.chunk_index + 1}\n\n")
                file_handle.write(f"```yaml\n{metadata_yaml}```\n\n")

                if item.title:
                    file_handle.write(f"**Title:** {item.title}\n\n")

                # Content block with language hint
                lang_hint = item.language if item.language and item.language not in ['unknown', 'en'] else ''
                file_handle.write(f"```{lang_hint}\n{item.chunk_text}\n```\n\n")
                file_handle.write("---\n\n")

            return True

        except Exception as e:
            self.logger.error(f"Markdown export failed: {e}")
            return False

    def _determine_export_path_and_format(self, first_item: RAGOutputItem, export_cfg: Optional[ExportConfig] = None) -> Tuple[str, str]:
        """Determine export path and format with intelligent defaults"""
        if export_cfg and export_cfg.output_path and export_cfg.format:
            path_str, format_str = export_cfg.output_path, export_cfg.format.lower()

            # Ensure directory exists
            output_dir = Path(path_str).parent
            if output_dir != Path('.'):
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                path_str = str(self.default_output_dir / Path(path_str).name)

            self.logger.info(f"Using configured export: {path_str} ({format_str})")
            return path_str, format_str

        # Generate intelligent default path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sanitize query and source type for filename
        query_safe = re.sub(r'[^\w\-_\.]', '_', first_item.query_used)[:30]
        source_safe = re.sub(r'[^\w\-_\.]', '_', first_item.source_type)[:20]

        # Create organized directory structure
        export_dir = self.default_output_dir / f"{query_safe}_{source_safe}"
        export_dir.mkdir(parents=True, exist_ok=True)

        path_str = str(export_dir / f"rag_export_{timestamp}.jsonl")
        format_str = "jsonl"

        self.logger.info(f"Using default export path: {path_str} ({format_str})")
        return path_str, format_str

def _clean_text_for_dedup(text: Optional[str]) -> str:
    """Clean text for deduplication"""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.lower()).strip()

def run_professional_pipeline(
        query_or_config_path: str,
        logger_instance=None,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        initial_content_type_hint: Optional[str] = None,
        max_retries: int = 3
) -> Tuple[List[EnrichedItem], PipelineMetrics]:
    """
    Professional-grade pipeline with comprehensive error handling, metrics, and recovery
    FIXED: Removed signal handling that was causing thread issues
    """

    # Initialize logger
    logger = logger_instance if logger_instance else logging.getLogger(config.DEFAULT_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    # Initialize metrics
    metrics = PipelineMetrics(start_time=datetime.now())

    def update_progress(step_name: str, current_step: int, total_steps: int = 10):
        """Enhanced progress reporting"""
        percentage = int((current_step / total_steps) * 100)
        if progress_callback:
            progress_callback(f"{step_name} ({current_step}/{total_steps})", percentage)
        logger.info(f"🔄 Pipeline progress: {step_name} - {percentage}%")

    try:
        # REMOVED: graceful_shutdown context manager - was causing signal issues in threads
        update_progress("Initializing Configuration", 1)

        # Initialize configuration manager
        cfg_manager = ConfigManager(logger_instance=logger)
        is_config_file_mode = os.path.exists(query_or_config_path) and query_or_config_path.lower().endswith(
            (".yaml", ".yml", ".json"))

        domain_query_for_log = query_or_config_path
        if is_config_file_mode:
            if not cfg_manager.load_config(query_or_config_path):
                metrics.errors.append("Failed to load configuration file")
                logger.error("❌ Configuration loading failed")
                return [], metrics

            domain_query_for_log = cfg_manager.config.domain_info.get('name',
                                                                      query_or_config_path) if cfg_manager.config else query_or_config_path

        logger.info(f"🚀 Professional pipeline starting: '{domain_query_for_log}'")

        update_progress("Initializing Components", 2)

        # Initialize components with error handling
        try:
            fetcher_pool = FetcherPool(
                num_workers=getattr(config, 'MAX_CONCURRENT_FETCHERS', 3),
                logger=logger
            )
            content_router = ContentRouter(config_manager=cfg_manager, logger_instance=logger)
            deduplicator = SmartDeduplicator(logger=logger)
            chunker = Chunker(logger_instance=logger)
            exporter = RobustExporter(logger=logger)
            quality_filter = ProfessionalQualityFilter(logger)
            enricher = ProfessionalContentEnricher(NLP_MODEL, logger)

        except Exception as e:
            metrics.errors.append(f"Component initialization failed: {e}")
            logger.error(f"❌ Component initialization failed: {e}", exc_info=True)
            return [], metrics

        update_progress("Preparing Fetch Tasks", 3)

        # Prepare fetch tasks
        current_run_export_config: Optional[ExportConfig] = None
        tasks_to_fetch = []

        if cfg_manager.config and cfg_manager.config.sources:
            # Configuration-based mode
            for src_cfg_model in cfg_manager.get_sources():
                if not current_run_export_config and src_cfg_model.export_config:
                    current_run_export_config = src_cfg_model.export_config

                for seed_url in src_cfg_model.seeds:
                    tasks_to_fetch.append(
                        (str(seed_url), src_cfg_model.source_type or src_cfg_model.name, domain_query_for_log,
                         None))

        elif not is_config_file_mode:
            # Query-based mode
            query_str = query_or_config_path

            # Set up default export configuration
            query_sanitized = re.sub(r'[^\w\-_\.]', '_', query_str)[:50]
            export_dir = Path("./data_exports") / f"query_{query_sanitized}"
            export_dir.mkdir(parents=True, exist_ok=True)
            current_run_export_config = ExportConfig(
                output_path=str(export_dir / "rag_export.jsonl"),
                format="jsonl"
            )

            if query_str.startswith(("http://", "https://")):
                # Direct URL
                tasks_to_fetch.append((query_str, "direct_url_query", query_str, None))
            elif DUCKDUCKGO_AVAILABLE:
                # Search query
                logger.info(f"🔍 Performing autonomous search: '{query_str}'")
                try:
                    with DDGS(timeout=10) as ddgs:
                        search_results = list(
                            ddgs.text(query_str, max_results=getattr(config, 'AUTONOMOUS_SEARCH_MAX_RESULTS', 5)))
                        for result in search_results:
                            if result.get('href'):
                                tasks_to_fetch.append(
                                    (result['href'], "autonomous_web_search", query_str, result.get('title')))
                except Exception as e:
                    metrics.errors.append(f"Search failed: {e}")
                    logger.error(f"❌ Search failed: {e}")
                    return [], metrics
            else:
                metrics.errors.append("No search capability available")
                logger.error("❌ Cannot perform autonomous search - DuckDuckGo not available")
                return [], metrics
        else:
            metrics.errors.append("No valid configuration or query provided")
            logger.error("❌ No valid configuration or query provided")
            return [], metrics

        if not tasks_to_fetch:
            metrics.errors.append("No URLs to fetch")
            logger.error("❌ No URLs prepared for fetching")
            return [], metrics

        metrics.total_urls = len(tasks_to_fetch)
        logger.info(f"📋 Prepared {len(tasks_to_fetch)} URLs for fetching")

        # Submit fetch tasks
        for url, source_type, query_used, item_title in tasks_to_fetch:
            fetcher_pool.submit_task(url, source_type, query_used, item_title)

        update_progress(f"Fetching Content ({len(tasks_to_fetch)} URLs)", 4)

        # Get fetch results
        fetched_items_all: List[FetchedItem] = fetcher_pool.get_results()
        metrics.successful_fetches = len(fetched_items_all)
        metrics.failed_fetches = metrics.total_urls - metrics.successful_fetches

        if not fetched_items_all:
            metrics.errors.append("No content fetched successfully")
            logger.warning("⚠️ No content was successfully fetched")
            fetcher_pool.shutdown()
            return [], metrics

        logger.info(f"✅ Fetched {len(fetched_items_all)} items (success rate: {metrics.success_rate:.1f}%)")

        update_progress("Parsing Content", 5)

        # Parse content
        parsed_items_all: List[ParsedItem] = []
        for item_fetched in fetched_items_all:
            if item_fetched.content_bytes or item_fetched.content:
                try:
                    parsed = content_router.route_and_parse(item_fetched)
                    if parsed:
                        parsed_items_all.append(parsed)
                except Exception as e:
                    metrics.errors.append(f"Parse error for {item_fetched.source_url}: {e}")
                    logger.warning(f"⚠️ Parse failed for {item_fetched.source_url}: {e}")

        metrics.parsed_items = len(parsed_items_all)
        logger.info(f"✅ Parsed {len(parsed_items_all)} items")

        update_progress("Normalizing & Deduplicating", 6)

        # Normalize and deduplicate
        normalized_items_all: List[NormalizedItem] = []
        for p_item in parsed_items_all:
            try:
                # Prepare content for deduplication
                full_text_parts = []
                if p_item.main_text_content:
                    full_text_parts.append(_clean_text_for_dedup(p_item.main_text_content))

                cleaned_structured_blocks = []
                for block_dict in p_item.extracted_structured_blocks:
                    content_to_clean = block_dict.get('content', '')
                    if block_dict.get('type') == 'semantic_figure_with_caption':
                        content_to_clean = f"{block_dict.get('figure_content', '')} {block_dict.get('caption_content', '')}".strip()

                    cleaned_content = _clean_text_for_dedup(content_to_clean)
                    if cleaned_content:
                        full_text_parts.append(cleaned_content)
                    cleaned_structured_blocks.append(block_dict.copy())

                # Check for duplicates
                full_content_signature = " ".join(filter(None, full_text_parts)).strip()
                if full_content_signature:
                    is_dup, _ = deduplicator.is_duplicate(full_content_signature)
                    if not is_dup:
                        deduplicator.add_snippet(full_content_signature)

                        # Create normalized item
                        norm_item = NormalizedItem(
                            id=p_item.id,
                            parsed_item_id=p_item.id,
                            source_url=p_item.source_url,
                            source_type=p_item.source_type,
                            query_used=p_item.query_used,
                            title=p_item.title,
                            cleaned_text_content=p_item.main_text_content,
                            cleaned_structured_blocks=cleaned_structured_blocks,
                            custom_fields=p_item.custom_fields,
                            language_of_main_text=p_item.detected_language_of_main_text
                        )
                        normalized_items_all.append(norm_item)
                    else:
                        metrics.duplicates_filtered += 1

            except Exception as e:
                metrics.errors.append(f"Normalization error for {p_item.source_url}: {e}")
                logger.warning(f"⚠️ Normalization failed for {p_item.source_url}: {e}")

        metrics.normalized_items = len(normalized_items_all)
        logger.info(
            f"✅ Normalized {len(normalized_items_all)} unique items (filtered {metrics.duplicates_filtered} duplicates)")

        update_progress("Quality Filtering", 7)

        # Apply quality filtering
        high_quality_items, filtered_count = quality_filter.filter_by_quality(normalized_items_all)
        metrics.quality_filtered = filtered_count

        update_progress("Enriching Metadata", 8)

        # Enrich content
        enriched_items_all: List[EnrichedItem] = []
        for n_item in high_quality_items:
            try:
                enriched_item = enricher.enrich_item(n_item)
                enriched_items_all.append(enriched_item)
            except Exception as e:
                metrics.errors.append(f"Enrichment error for {n_item.source_url}: {e}")
                logger.warning(f"⚠️ Enrichment failed for {n_item.source_url}: {e}")
                # Add fallback enriched item
                enriched_items_all.append(enricher._create_fallback_enriched_item(n_item))

        metrics.enriched_items = len(enriched_items_all)
        logger.info(f"✅ Enriched {len(enriched_items_all)} items")

        # Store enhanced data for GUI
        if hasattr(logger, 'enhanced_snippet_data'):
            logger.enhanced_snippet_data = [item.displayable_metadata_summary for item in enriched_items_all]

        update_progress("Chunking & Formatting for RAG", 9)

        # Create RAG chunks
        all_rag_chunks: List[RAGOutputItem] = []
        for e_item in enriched_items_all:
            try:
                chunks = chunker.chunk_item(e_item)
                all_rag_chunks.extend(chunks)
            except Exception as e:
                metrics.errors.append(f"Chunking error for {e_item.source_url}: {e}")
                logger.warning(f"⚠️ Chunking failed for {e_item.source_url}: {e}")

        metrics.rag_chunks = len(all_rag_chunks)
        logger.info(f"✅ Generated {len(all_rag_chunks)} RAG chunks")

        update_progress("Exporting RAG Data", 10)

        # Export RAG chunks
        if all_rag_chunks:
            export_success = exporter.export_batch(all_rag_chunks, export_cfg=current_run_export_config)
            if not export_success:
                metrics.errors.append("Export failed")
                logger.error("❌ RAG export failed")
        else:
            logger.info("ℹ️ No RAG chunks to export")

        # Cleanup
        fetcher_pool.shutdown()

        # Finalize metrics
        metrics.end_time = datetime.now()

        # Log final summary
        logger.info(f"""
🎉 Professional pipeline completed successfully!
📊 Summary:
   • Duration: {metrics.duration.total_seconds():.1f}s
   • Success Rate: {metrics.success_rate:.1f}%
   • Items Processed: {metrics.enriched_items}
   • RAG Chunks: {metrics.rag_chunks}
   • Errors: {len(metrics.errors)}
""")

        if metrics.errors:
            logger.warning(f"⚠️ {len(metrics.errors)} errors occurred during processing")
            for i, error in enumerate(metrics.errors[:5], 1):  # Show first 5 errors
                logger.warning(f"  {i}. {error}")
            if len(metrics.errors) > 5:
                logger.warning(f"  ... and {len(metrics.errors) - 5} more errors")

        return enriched_items_all, metrics

    except KeyboardInterrupt:
        logger.warning("🛑 Pipeline interrupted by user")
        metrics.errors.append("Pipeline interrupted by user")
        metrics.end_time = datetime.now()
        return [], metrics

    except Exception as e:
        logger.error(f"💥 Pipeline failed with critical error: {e}", exc_info=True)
        metrics.errors.append(f"Critical pipeline failure: {e}")
        metrics.end_time = datetime.now()
        return [], metrics


def search_and_fetch(
        query_or_config_path: str,
        logger: logging.Logger,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        content_type_gui: Optional[str] = None
) -> List[EnrichedItem]:
    """
    Main entry point for professional scraping pipeline
    Returns enriched items for backward compatibility
    """
    logger.info(f"🎯 Professional search initiated: '{query_or_config_path}' (hint: {content_type_gui})")

    enriched_items, metrics = run_professional_pipeline(
        query_or_config_path,
        logger_instance=logger,
        progress_callback=progress_callback,
        initial_content_type_hint=content_type_gui
    )

    # Log metrics summary
    metrics_summary = metrics.to_dict()
    logger.info(f"📈 Pipeline metrics: {json.dumps(metrics_summary, indent=2)}")

    if progress_callback:
        progress_callback("Processing complete!", 100)

    return enriched_items


# Legacy compatibility functions
def fetch_stdlib_docs(m, l):
    return []


def fetch_stackoverflow_snippets(q, l, t=None):
    return [], []


def fetch_github_readme_snippets(q, l, mr=None, spr=None):
    return []


def fetch_github_file_snippets(q, l, mr=None, fprt=None):
    return [], []


def detect_content_type(q: str, l):
    return config.DEFAULT_CONTENT_TYPE_FOR_GUI