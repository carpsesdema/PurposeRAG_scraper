# scraper/rag_models.py
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, HttpUrl, Field
import uuid
from datetime import datetime


class FetchedItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_url: HttpUrl
    content: Optional[str] = None
    content_bytes: Optional[bytes] = None
    content_type_detected: Optional[str] = None
    source_type: str
    query_used: str
    title: Optional[str] = None
    depth: int = 0
    encoding: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractedLinkInfo(BaseModel):
    url: HttpUrl
    text: Optional[str] = None
    rel: Optional[str] = None


class ParsedItem(BaseModel):
    id: str
    fetched_item_id: str
    source_url: HttpUrl
    source_type: str  # User-defined type for the source (e.g., 'tennis_article', 'player_bio')
    query_used: str

    title: Optional[str] = None
    main_text_content: Optional[str] = None
    extracted_structured_blocks: List[Dict[str, Any]] = Field(default_factory=list)

    custom_fields: Dict[str, Any] = Field(default_factory=dict,
                                          description="Key-value pairs of specifically extracted data points from YAML config.")  # <-- NEW

    detected_language_of_main_text: Optional[str] = None
    extracted_links: List[ExtractedLinkInfo] = Field(default_factory=list)
    parser_metadata: Dict[str, Any] = Field(default_factory=dict)


class NormalizedItem(BaseModel):
    id: str
    parsed_item_id: str
    source_url: HttpUrl
    source_type: str
    query_used: str

    title: Optional[str] = None
    cleaned_text_content: Optional[str] = None
    cleaned_structured_blocks: List[Dict[str, Any]] = Field(default_factory=list)

    custom_fields: Dict[str, Any] = Field(default_factory=dict)  # <-- NEW (propagated)

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

    primary_text_content: Optional[str] = None
    enriched_structured_elements: List[Dict[str, Any]] = Field(default_factory=list)

    custom_fields: Dict[str, Any] = Field(default_factory=dict)  # <-- NEW (propagated)

    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    keyphrases: List[str] = Field(default_factory=list)
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
    chunk_index: int
    chunk_parent_type: str = Field(default="primary_text")
    chunk_parent_element_index: Optional[int] = None
    total_chunks_for_parent_element: int

    title: Optional[str] = None
    language: Optional[str] = None

    # Metadata propagated from EnrichedItem or specific to chunk
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    keyphrases: List[str] = Field(default_factory=list)
    entities_in_chunk: List[Dict[str, str]] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)  # <-- NEW (propagated to chunks)

    quality_score: Optional[float] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")