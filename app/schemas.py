# app/schemas.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi import UploadFile

class ResultItem(BaseModel):
    id: str
    content_type: Optional[str]
    image_url: Optional[str]
    score: float

class SearchResultSchema(BaseModel):
    results: List[ResultItem]


class AdvancedSearchFilters(BaseModel):
    style: Optional[str] = None
    palette_contains: Optional[str] = None  # HEX
    block: Optional[str] = None
    device: Optional[str] = None
    categories_contains: Optional[str] = None  # tag
    mode: Optional[str] = None  # dark/light/mixed


class AdvancedSearchResponseItem(ResultItem):
    extra: Optional[Dict[str, Any]] = None


class AdvancedSearchResponse(BaseModel):
    results: List[AdvancedSearchResponseItem]

class UnifiedSearchRequest(BaseModel):
    query_text: Optional[str] = None
    top_k: int = 10
    
    class Config:
        # Allow arbitrary types for file uploads
        arbitrary_types_allowed = True
