from typing import List
from .rag_models import EnrichedItem, RAGOutputItem
from config import DEFAULT_CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE


class Chunker:
    def __init__(self, logger,
                 max_chunk_size_chars: int = DEFAULT_CHUNK_SIZE,
                 min_chunk_size_chars: int = MIN_CHUNK_SIZE,
                 overlap_size_chars: int = CHUNK_OVERLAP):
        self.logger = logger
        self.max_chunk_size = max_chunk_size_chars
        self.min_chunk_size = min_chunk_size_chars
        self.overlap_size = overlap_size_chars

    def _chunk_text_content(self, text: str, enriched_item: EnrichedItem, base_idx: int = 0) -> List[RAGOutputItem]:
        """Chunks plain text content."""
        if not text or len(text.strip()) < self.min_chunk_size:
            return []

        # Naive chunking by character count for now.
        # A more sophisticated approach would use sentence splitting or semantic chunking.
        text_chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chunk_size, len(text))
            chunk = text[start:end]
            text_chunks.append(chunk)
            start = end - self.overlap_size if end < len(text) else end  # Apply overlap

        output_items = []
        for i, chunk_content in enumerate(text_chunks):
            if len(chunk_content.strip()) < self.min_chunk_size:
                continue

            rag_item = RAGOutputItem(
                parent_item_id=enriched_item.id,
                source_url=enriched_item.source_url,
                source_type=enriched_item.source_type,
                query_used=enriched_item.query_used,
                chunk_text=chunk_content.strip(),
                chunk_index=base_idx + i,
                total_chunks=len(text_chunks),  # This is total for this specific text_content, not overall
                title=enriched_item.title,
                language=enriched_item.detailed_enrichment_data_for_gui[0].get(
                    'content_type') if enriched_item.detailed_enrichment_data_for_gui else 'en',
                # Default to 'en' for text
                categories=enriched_item.categories,
                tags=enriched_item.tags,
                quality_score=enriched_item.quality_score,
                complexity=enriched_item.complexity
            )
            output_items.append(rag_item)
        return output_items

    def _chunk_code_item(self, code_data: dict, enriched_item: EnrichedItem, item_idx: int, base_chunk_idx: int) -> \
    List[RAGOutputItem]:
        """Chunks a single code item (dictionary with 'code' and 'language')."""
        code_text = code_data.get('code', '')
        language = code_data.get('language', 'unknown')  # Default if language not specified

        if not code_text or len(code_text.strip()) < self.min_chunk_size:
            return []

        # Chunk code by lines, trying to respect max_chunk_size
        # More advanced: use AST nodes for semantic code chunking
        lines = code_text.splitlines()
        current_code_chunks_content = []
        current_chunk_lines = []
        current_len = 0

        for line in lines:
            if current_len + len(line) + 1 > self.max_chunk_size and current_chunk_lines:
                current_code_chunks_content.append("\n".join(current_chunk_lines))
                # Overlap for code: could be previous N lines or based on indentation
                overlap_n_lines = max(1, self.overlap_size // 80)  # Approx lines based on avg line length
                current_chunk_lines = current_chunk_lines[-overlap_n_lines:] if self.overlap_size > 0 else []
                current_chunk_lines.append(line)
                current_len = sum(len(l) + 1 for l in current_chunk_lines)
            else:
                current_chunk_lines.append(line)
                current_len += len(line) + 1

        if current_chunk_lines:
            current_code_chunks_content.append("\n".join(current_chunk_lines))

        output_items = []
        for i, chunk_content in enumerate(current_code_chunks_content):
            if len(chunk_content.strip()) < self.min_chunk_size:
                continue

            # Metadata for this chunk might come from the corresponding entry in detailed_enrichment_data_for_gui
            specific_enrichment = {}
            if item_idx < len(enriched_item.detailed_enrichment_data_for_gui):
                specific_enrichment = enriched_item.detailed_enrichment_data_for_gui[item_idx]

            rag_item = RAGOutputItem(
                parent_item_id=enriched_item.id,
                source_url=enriched_item.source_url,
                source_type=enriched_item.source_type,
                query_used=enriched_item.query_used,
                chunk_text=chunk_content.strip(),
                chunk_index=base_chunk_idx + i,
                total_chunks=len(current_code_chunks_content),  # Total for this specific code_item
                title=enriched_item.title,
                language=language,
                categories=specific_enrichment.get('categories', enriched_item.categories),
                tags=specific_enrichment.get('tags', enriched_item.tags),
                quality_score=specific_enrichment.get('quality_score', enriched_item.quality_score),
                complexity=specific_enrichment.get('complexity', enriched_item.complexity)
            )
            output_items.append(rag_item)
        return output_items

    def chunk_item(self, enriched_item: EnrichedItem) -> List[RAGOutputItem]:
        rag_outputs = []
        current_chunk_offset = 0

        # Process main text_content if available
        if enriched_item.text_content:
            self.logger.debug(f"Chunking text_content from {enriched_item.source_url}")
            text_chunks = self._chunk_text_content(enriched_item.text_content, enriched_item,
                                                   base_idx=current_chunk_offset)
            rag_outputs.extend(text_chunks)
            current_chunk_offset += len(text_chunks)

        # Process code items
        for idx, code_data in enumerate(enriched_item.code_items):
            self.logger.debug(f"Chunking code_item {idx} from {enriched_item.source_url}")
            code_chunks = self._chunk_code_item(code_data, enriched_item, idx, base_chunk_idx=current_chunk_offset)
            rag_outputs.extend(code_chunks)
            current_chunk_offset += len(code_chunks)

        if not rag_outputs:
            self.logger.warning(
                f"No RAG output items generated for enriched item ID {enriched_item.id} from {enriched_item.source_url}. "
                f"Text content present: {bool(enriched_item.text_content)}, "
                f"Code items present: {len(enriched_item.code_items)}"
            )
        else:
            # Update total_chunks for all items now that we know the grand total for this EnrichedItem
            grand_total_chunks = len(rag_outputs)
            for item in rag_outputs:
                item.total_chunks = grand_total_chunks

        return rag_outputs