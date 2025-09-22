# app/schemas.py
from pydantic import BaseModel
from typing import List, Optional
from fastapi import UploadFile

class ResultItem(BaseModel):
    id: str
    content_type: Optional[str]
    image_url: Optional[str]
    score: float

class SearchResultSchema(BaseModel):
    results: List[ResultItem]

class UnifiedSearchRequest(BaseModel):
    query_text: Optional[str] = None
    top_k: int = 10
    
    class Config:
        # Allow arbitrary types for file uploads
        arbitrary_types_allowed = True
