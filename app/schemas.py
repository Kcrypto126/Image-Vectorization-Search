# app/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class ResultItem(BaseModel):
    id: str
    content_type: Optional[str]
    image_url: Optional[str]
    score: float

class SearchResultSchema(BaseModel):
    results: List[ResultItem]
