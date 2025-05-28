# scraper/chunker.py
import logging
from typing import List, Dict, Optional, Any
from .rag_models import EnrichedItem, RAGOutputItem
from config import DEFAULT_CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE

# Fixed logger import - use standard logging instead of multiprocessing.get_logger
logger = logging.getLogger(__name__)


class Chunker:
    def __init__(self, logger_instance=None,
                 max_chunk_size_chars: int = DEFAULT_CHUNK_SIZE,
                 min_chunk_size_chars: int = MIN_CHUNK_SIZE,
                 overlap_size_chars: int = CHUNK_OVERLAP):
        self.logger = logger_instance if logger_instance else logger
        self.max_chunk_size = max_chunk_size_chars
        self.min_chunk_size = min_chunk_size_chars
        self.overlap_size = overlap_size_chars

    def _create_rag_item(self, enriched_item: EnrichedItem, chunk_text: str,
                         chunk_idx: int, total_chunks_for_elem: int,
                         parent_type: str, parent_element_idx: Optional[int] = None,
                         element_specific_lang: Optional[str] = None,
                         element_specific_entities: Optional[List[Dict[str, str]]] = None
                         ) -> RAGOutputItem:

        chunk_keyphrases = enriched_item.keyphrases
        # Propagate all custom_fields from the parent EnrichedItem to each chunk
        chunk_custom_fields = enriched_item.custom_fields

        return RAGOutputItem(
            parent_item_id=enriched_item.id,
            source_url=enriched_item.source_url,
            source_type=enriched_item.source_type,
            query_used=enriched_item.query_used,
            chunk_text=chunk_text.strip(),
            chunk_index=chunk_idx,
            chunk_parent_type=parent_type,
            chunk_parent_element_index=parent_element_idx,
            total_chunks_for_parent_element=total_chunks_for_elem,
            title=enriched_item.title,
            language=element_specific_lang or enriched_item.language_of_primary_text,
            categories=enriched_item.categories,
            tags=enriched_item.tags,
            keyphrases=chunk_keyphrases,
            custom_fields=chunk_custom_fields,  # <-- Propagate custom_fields
            entities_in_chunk=element_specific_entities if element_specific_entities is not None else enriched_item.overall_entities[
                                                                                                      :5],
            quality_score=enriched_item.quality_score,
        )

    def _chunk_single_content_piece(self, content: str, enriched_item: EnrichedItem,
                                    parent_type: str, parent_element_idx: Optional[int] = None,
                                    element_lang: Optional[str] = None,
                                    element_entities: Optional[List[Dict[str, str]]] = None
                                    ) -> List[RAGOutputItem]:
        # ... (Chunking logic for a single piece remains the same as your last provided version) ...
        if not content or len(content.strip()) < self.min_chunk_size:
            self.logger.debug(
                f"Skipping chunking for content piece (type: {parent_type}, parent_idx: {parent_element_idx}) due to insufficient length or no content.")
            return []
        text_chunks_content = []
        start = 0;
        text_length = len(content)
        while start < text_length:
            end = min(start + self.max_chunk_size, text_length)
            chunk = content[start:end];
            text_chunks_content.append(chunk)
            if end == text_length: break
            start = max(0, end - self.overlap_size)
            if len(text_chunks_content) > text_length / (
            self.min_chunk_size / 2 if self.min_chunk_size > 0 else 10) + 10: self.logger.warning(
                f"Potential infinite loop in chunking for {enriched_item.source_url}, type {parent_type}. Breaking."); break
        output_items = []
        for i, chunk_content_str in enumerate(text_chunks_content):
            stripped_chunk = chunk_content_str.strip()
            if len(stripped_chunk) < self.min_chunk_size: continue
            chunk_specific_entities = []
            rag_item = self._create_rag_item(enriched_item, stripped_chunk, chunk_idx=i,
                                             total_chunks_for_elem=len(text_chunks_content), parent_type=parent_type,
                                             parent_element_idx=parent_element_idx, element_specific_lang=element_lang,
                                             element_specific_entities=chunk_specific_entities)
            output_items.append(rag_item)
        return output_items

    def chunk_item(self, enriched_item: EnrichedItem) -> List[RAGOutputItem]:
        # ... (Logic for deciding what to chunk remains the same as your last provided version) ...
        rag_outputs = []
        if enriched_item.primary_text_content:
            self.logger.debug(
                f"Chunking primary_text_content from {enriched_item.source_url} ({len(enriched_item.primary_text_content)} chars)")
            text_chunks = self._chunk_single_content_piece(enriched_item.primary_text_content, enriched_item,
                                                           parent_type="primary_text",
                                                           element_lang=enriched_item.language_of_primary_text,
                                                           element_entities=enriched_item.overall_entities)
            rag_outputs.extend(text_chunks)
            self.logger.debug(f"Generated {len(text_chunks)} chunks from primary text of {enriched_item.source_url}")
        for idx, structured_element in enumerate(enriched_item.enriched_structured_elements):
            element_content_to_chunk: Optional[str] = None;
            element_type = structured_element.get('type', 'unknown_structured_element')
            if element_type == 'semantic_figure_with_caption':
                element_content_to_chunk = f"{structured_element.get('figure_content', '')} {structured_element.get('caption_content', '')}".strip()
            else:
                element_content_to_chunk = structured_element.get('content')
            element_lang_hint = structured_element.get('language');
            element_specific_entities = structured_element.get('entities')
            self.logger.debug(
                f"Chunking structured_element {idx} (type: {element_type}) from {enriched_item.source_url} ({len(element_content_to_chunk or '')} chars)")
            if element_content_to_chunk:
                element_chunks = self._chunk_single_content_piece(element_content_to_chunk, enriched_item,
                                                                  parent_type=element_type, parent_element_idx=idx,
                                                                  element_lang=element_lang_hint,
                                                                  element_entities=element_specific_entities)
                rag_outputs.extend(element_chunks)
                self.logger.debug(
                    f"Generated {len(element_chunks)} chunks from structured element {idx} (type: {element_type}) of {enriched_item.source_url}")
        if not rag_outputs: self.logger.warning(
            f"No RAG output items generated for enriched item ID {enriched_item.id} from {enriched_item.source_url}.")
        return rag_outputs