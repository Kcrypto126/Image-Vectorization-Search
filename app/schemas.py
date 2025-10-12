# app/schemas.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi import UploadFile

class ResultItem(BaseModel):
    id: str
    content_type: Optional[str]
    image_url: Optional[str]
    score: float
    taste_bonus: Optional[float] = None
    topic_bonus: Optional[float] = None

class SearchResultSchema(BaseModel):
    results: List[ResultItem]

class UserTasteProfile(BaseModel):
    user_id: str
    taste_vector: Optional[List[float]]
    top_topics: List[Dict[str, Any]]
    interaction_count: int
    boards: List[Dict[str, str]]
    last_updated: str

class BoardSchema(BaseModel):
    board_id: str
    user_id: str
    name: str
    description: Optional[str]
    status: str

class PinBoardSchema(BaseModel):
    pin_id: str
    board_id: str
    status: str

class TopicSchema(BaseModel):
    topic_id: str
    name: str
    description: Optional[str]
    confidence: Optional[float] = None

class UserTopicAffinity(BaseModel):
    topic_id: str
    affinity_score: float
    interaction_count: int
    last_interaction: Optional[str]

class TasteGraphStatus(BaseModel):
    status: str
    nodes: Optional[int] = None
    edges: Optional[int] = None
    users_trained: Optional[int] = None
    epochs: Optional[int] = None
    final_loss: Optional[float] = None
    error: Optional[str] = None
