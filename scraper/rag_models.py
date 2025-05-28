from typing import List, Dict, Optional, Any, Literal
from pydantic import BaseModel, HttpUrl, validator
import uuid
from datetime import datetime


class FetchedItem(BaseModel):
    """Data structure for content successfully fetched."""
    id: str
    source_url: HttpUrl
    content: str
    content_type_detected: Optional[str] = None
    source_type: str
    query_used: str
    depth: int = 0
    metadata: Dict[str, Any] = {}

    @classmethod
    def create(cls, source_url: HttpUrl, content: str, source_type: str, query_used: str, **kwargs):
        return cls(id=str(uuid.uuid4()), source_url=source_url, content=content,
                   source_type=source_type, query_used=query_used, **kwargs)


class ParsedItem(BaseModel):
    """Data structure for content after parsing and initial extraction."""
    id: str
    fetched_item_id: str
    source_url: HttpUrl
    source_type: str
    query_used: str

    title: Optional[str] = None
    main_text_content: Optional[str] = None
    code_snippets: List[str] = []

    detected_language: Optional[str] = None

    extracted_links: List[HttpUrl] = []

    parser_metadata: Dict[str, Any] = {}


class NormalizedItem(BaseModel):
    """Data after cleaning, normalization, and deduplication."""
    id: str
    parsed_item_id: str
    source_url: HttpUrl
    source_type: str
    query_used: str

    title: Optional[str] = None
    cleaned_text: Optional[str] = None
    cleaned_code_snippets: List[Dict[str, Any]] = []

    is_duplicate: bool = False
    normalization_metadata: Dict[str, Any] = {}


class EnrichedItem(BaseModel):
    """Data after metadata enrichment (categorization, NER, etc.)."""
    id: str
    normalized_item_id: str
    source_url: HttpUrl
    source_type: str
    query_used: str
    title: Optional[str] = None

    text_content: Optional[str] = None
    code_items: List[Dict[str, Any]] = []

    categories: List[str] = []
    tags: List[str] = []
    quality_score: Optional[float] = None
    freelance_score: Optional[float] = None
    trading_value_score: Optional[float] = None
    complexity: Optional[str] = None
    entities: List[str] = []
    use_cases: List[str] = []

    detailed_enrichment_data_for_gui: List[Dict[str, Any]] = []


class RAGOutputItem(BaseModel):
    """Final RAG-ready item, potentially a chunk."""
    id: str = ""
    parent_item_id: str
    source_url: HttpUrl
    source_type: str
    query_used: str

    chunk_text: str
    chunk_index: int
    total_chunks: int

    title: Optional[str] = None
    language: Optional[str] = None

    categories: List[str] = []
    tags: List[str] = []
    quality_score: Optional[float] = None
    complexity: Optional[str] = None
    timestamp: str = ""

    @validator('id', pre=True, always=True)
    def set_id(cls, v):
        return v or str(uuid.uuid4())

    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls, v):
        return v or datetime.utcnow().isoformat() + "Z"