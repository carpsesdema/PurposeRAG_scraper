from typing import List, Dict, Optional, Any
from pydantic import BaseModel, HttpUrl, validator, Field
import uuid
from datetime import datetime


class FetchedItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_url: HttpUrl
    content: str
    content_type_detected: Optional[str] = None
    source_type: str
    query_used: str
    depth: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParsedItem(BaseModel):
    id: str
    fetched_item_id: str
    source_url: HttpUrl
    source_type: str
    query_used: str

    title: Optional[str] = None
    main_text_content: Optional[str] = None

    extracted_structured_blocks: List[Dict[str, str]] = Field(default_factory=list,
                                                              description="e.g., [{'type': 'table_markdown', 'content': '...'}, {'type': 'fenced_block', 'language_hint': 'json', 'content': '...'}]")

    detected_language_of_main_text: Optional[str] = None
    extracted_links: List[HttpUrl] = Field(default_factory=list)
    parser_metadata: Dict[str, Any] = Field(default_factory=dict)


class NormalizedItem(BaseModel):
    id: str
    parsed_item_id: str
    source_url: HttpUrl
    source_type: str
    query_used: str

    title: Optional[str] = None
    cleaned_text_content: Optional[str] = None

    # Stores the cleaned structured blocks from ParsedItem
    cleaned_structured_blocks: List[Dict[str, str]] = Field(default_factory=list)

    is_duplicate: bool = False
    normalization_metadata: Dict[str, Any] = Field(default_factory=dict)
    language_of_main_text: Optional[str] = None


class EnrichedItem(BaseModel):
    id: str
    normalized_item_id: str
    source_url: HttpUrl
    source_type: str
    query_used: str
    title: Optional[str] = None

    primary_text_content: Optional[str] = None  # Main narrative text

    # Enriched structured elements, retaining their type and content
    enriched_structured_elements: List[Dict[str, Any]] = Field(default_factory=list,
                                                               description="Each dict could be {'type': 'block_type', 'content': '...', 'language': 'optional_lang', 'entities': [], ...}")

    # Overall/summary metadata for the item
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    # Combined entities from primary_text and structured_elements, or just from primary_text
    overall_entities: List[Dict[str, str]] = Field(default_factory=list)
    language_of_primary_text: Optional[str] = None
    quality_score: Optional[float] = None
    complexity_score: Optional[float] = None

    displayable_metadata_summary: Dict[str, Any] = Field(default_factory=dict)


class RAGOutputItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_item_id: str
    source_url: HttpUrl
    source_type: str
    query_used: str

    chunk_text: str
    chunk_index: int  # Index of this chunk within its specific parent element (e.g., main text or one structured block)
    chunk_parent_type: str = Field(default="main_text",
                                   description="Indicates if chunk is from 'main_text' or a 'structured_element_type'")
    chunk_parent_element_index: Optional[int] = None  # If from structured_elements list, its index
    total_chunks_for_parent_element: int  # Total chunks for the specific element this chunk belongs to

    title: Optional[str] = None
    language: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    entities_in_chunk: List[Dict[str, str]] = Field(default_factory=list)
    quality_score: Optional[float] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")