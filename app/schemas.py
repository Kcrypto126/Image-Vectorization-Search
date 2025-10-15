# app/schemas.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi import UploadFile

class ResultItem(BaseModel):
    id: str
    content_type: Optional[str]
    image_url: Optional[str]
    url: Optional[str]  # Alternative field name for compatibility
    score: float
    taste_bonus: Optional[float] = None
    topic_bonus: Optional[float] = None
    board_bonus: Optional[float] = None
    collaborative_bonus: Optional[float] = None
    behavioral_bonus: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class SearchResultSchema(BaseModel):
    results: List[ResultItem]
    total: Optional[int] = None
    query: Optional[str] = None
    user_id: Optional[str] = None
    personalized: Optional[bool] = None
    error: Optional[str] = None

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
    board_type: str = "user"  # "user" or "auto"
    pin_count: int = 0
    created_at: Optional[str] = None
    last_activity: Optional[str] = None
    status: str

class PinBoardSchema(BaseModel):
    pin_id: str
    board_id: str
    status: str

# AutoBoardSchema removed - auto-boards are now managed automatically by the scheduler

class RecommendationSchema(BaseModel):
    pin_id: str
    score: float
    reason: str

class UserBoardsSchema(BaseModel):
    user_id: str
    boards: List[BoardSchema]
    count: int

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
