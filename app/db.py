# app/db.py
from sqlalchemy import create_engine, Column, String, DateTime, JSON, inspect, text, Integer, Float, Index, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os
import uuid
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

class ImageMetadata(Base):
    __tablename__ = "images"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    content_type = Column(String, nullable=True)
    image_url = Column(String, nullable=False)
    # Use different attribute and column name to avoid SQLAlchemy reserved word conflict
    image_metadata = Column('image_metadata', JSON, nullable=True)
    # New fields
    raw_detections = Column(JSON, nullable=True)
    generated_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Interaction(Base):
    __tablename__ = "interactions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)
    image_id = Column(String, nullable=False, index=True)
    event = Column(String, nullable=False)  # like, click, save, view
    weight = Column(Float, nullable=False, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

Index('ix_interactions_user_image', Interaction.user_id, Interaction.image_id)

class ItemCooccurrence(Base):
    __tablename__ = "item_cooccurrence"
    id = Column(Integer, primary_key=True, autoincrement=True)
    item_a = Column(String, nullable=False, index=True)
    item_b = Column(String, nullable=False, index=True)
    score = Column(Float, nullable=False, default=0.0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

Index('ix_item_pair', ItemCooccurrence.item_a, ItemCooccurrence.item_b, unique=True)

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, unique=True, index=True)  # External user ID
    taste_vector = Column(JSON, nullable=True)  # User's taste embedding vector
    taste_vector_version = Column(Integer, default=1)  # Version for tracking updates
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class Board(Base):
    __tablename__ = "boards"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    board_id = Column(String, nullable=False, unique=True, index=True)  # External board ID
    user_id = Column(String, nullable=False, index=True)  # Owner of the board
    name = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    board_vector = Column(JSON, nullable=True)  # Board's aggregated embedding
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PinBoard(Base):
    __tablename__ = "pin_boards"
    id = Column(Integer, primary_key=True, autoincrement=True)
    pin_id = Column(String, nullable=False, index=True)  # References ImageMetadata.id
    board_id = Column(String, nullable=False, index=True)  # References Board.board_id
    created_at = Column(DateTime, default=datetime.utcnow)

Index('ix_pin_board', PinBoard.pin_id, PinBoard.board_id, unique=True)

class Topic(Base):
    __tablename__ = "topics"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    topic_id = Column(String, nullable=False, unique=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    topic_vector = Column(JSON, nullable=True)  # Topic's embedding vector
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PinTopic(Base):
    __tablename__ = "pin_topics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    pin_id = Column(String, nullable=False, index=True)  # References ImageMetadata.id
    topic_id = Column(String, nullable=False, index=True)  # References Topic.topic_id
    confidence = Column(Float, default=1.0)  # Confidence score for the association
    created_at = Column(DateTime, default=datetime.utcnow)

Index('ix_pin_topic', PinTopic.pin_id, PinTopic.topic_id, unique=True)

class UserTopic(Base):
    __tablename__ = "user_topics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, nullable=False, index=True)  # References User.user_id
    topic_id = Column(String, nullable=False, index=True)  # References Topic.topic_id
    affinity_score = Column(Float, default=0.0)  # User's affinity to this topic
    interaction_count = Column(Integer, default=0)  # Number of interactions with this topic
    last_interaction = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

Index('ix_user_topic', UserTopic.user_id, UserTopic.topic_id, unique=True)

class TasteGraphEdge(Base):
    __tablename__ = "taste_graph_edges"
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_type = Column(String, nullable=False)  # 'user', 'pin', 'board', 'topic'
    source_id = Column(String, nullable=False, index=True)
    target_type = Column(String, nullable=False)  # 'user', 'pin', 'board', 'topic'
    target_id = Column(String, nullable=False, index=True)
    edge_type = Column(String, nullable=False)  # 'interaction', 'similarity', 'cooccurrence', 'topic_affinity'
    weight = Column(Float, default=1.0)
    edge_data = Column(JSON, nullable=True)  # Additional edge metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

Index('ix_taste_edge_source', TasteGraphEdge.source_type, TasteGraphEdge.source_id)
Index('ix_taste_edge_target', TasteGraphEdge.target_type, TasteGraphEdge.target_id)
Index('ix_taste_edge_type', TasteGraphEdge.edge_type)

def init_db(database_url: str = None):
    if database_url:
        global engine, SessionLocal
        engine = create_engine(database_url, pool_pre_ping=True)
        SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    Base.metadata.create_all(engine)
    # Lightweight auto-migration for adding new JSON columns if the table already exists
    try:
        inspector = inspect(engine)
        cols = {c['name'] for c in inspector.get_columns('images')}
        with engine.begin() as conn:
            # Handle migration from old 'metadata' column to 'image_metadata'
            if 'metadata' in cols and 'image_metadata' not in cols:
                try:
                    conn.execute(text("ALTER TABLE images RENAME COLUMN metadata TO image_metadata"))
                except Exception:
                    pass
            elif 'image_metadata' not in cols:
                try:
                    conn.execute(text("ALTER TABLE images ADD COLUMN image_metadata JSON"))
                except Exception:
                    pass
            if 'raw_detections' not in cols:
                try:
                    conn.execute(text("ALTER TABLE images ADD COLUMN raw_detections JSON"))
                except Exception:
                    pass
            if 'generated_metadata' not in cols:
                try:
                    conn.execute(text("ALTER TABLE images ADD COLUMN generated_metadata JSON"))
                except Exception:
                    pass
    except Exception:
        # Best-effort; ignore migration errors in lightweight setup
        pass

class DBSession:
    def __init__(self):
        self.session = SessionLocal()

    def add_image(self, id_, content_type, image_url, metadata=None, raw_detections=None, generated_metadata=None):
        img = ImageMetadata(
            id=id_,
            content_type=content_type,
            image_url=image_url,
            image_metadata=metadata,
            raw_detections=raw_detections,
            generated_metadata=generated_metadata,
        )
        self.session.add(img)
        self.session.commit()

    def get_image_metadata(self, id_):
        return self.session.get(ImageMetadata, id_)

    # Taste graph helpers
    def record_interaction(self, user_id: str, image_id: str, event: str, weight: float):
        it = Interaction(user_id=user_id, image_id=image_id, event=event, weight=weight)
        self.session.add(it)
        self.session.commit()
        return it

    def get_user_interactions(self, user_id: str):
        return self.session.query(Interaction).filter(Interaction.user_id == user_id).all()

    def upsert_cooccurrence(self, item_a: str, item_b: str, score: float):
        if not item_a or not item_b or item_a == item_b:
            return
        # Ensure deterministic order to avoid duplicates
        a, b = (item_a, item_b) if item_a < item_b else (item_b, item_a)
        existing = self.session.query(ItemCooccurrence).filter(
            ItemCooccurrence.item_a == a, ItemCooccurrence.item_b == b
        ).first()
        if existing:
            existing.score = float(score)
        else:
            self.session.add(ItemCooccurrence(item_a=a, item_b=b, score=float(score)))
        self.session.commit()

    def get_neighbors(self, item_id: str, top_k: int = 50):
        # Return neighbors for both (item_id, *) regardless of order
        rows = self.session.query(ItemCooccurrence).filter(
            (ItemCooccurrence.item_a == item_id) | (ItemCooccurrence.item_b == item_id)
        ).all()
        neighbors = []
        for r in rows:
            other = r.item_b if r.item_a == item_id else r.item_a
            neighbors.append((other, float(r.score)))
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:top_k]

    # Taste Graph methods
    def get_or_create_user(self, user_id: str):
        """Get existing user or create new one with initialized taste vector"""
        user = self.session.query(User).filter(User.user_id == user_id).first()
        if not user:
            user = User(user_id=user_id)
            self.session.add(user)
            self.session.commit()
        return user

    def update_user_taste_vector(self, user_id: str, taste_vector: np.ndarray, version: int = None):
        """Update user's taste vector"""
        user = self.get_or_create_user(user_id)
        user.taste_vector = taste_vector.tolist()
        if version:
            user.taste_vector_version = version
        else:
            user.taste_vector_version += 1
        user.updated_at = datetime.utcnow()
        self.session.commit()
        return user

    def get_user_taste_vector(self, user_id: str):
        """Get user's current taste vector"""
        user = self.session.query(User).filter(User.user_id == user_id).first()
        if user and user.taste_vector:
            return np.array(user.taste_vector)
        return None

    def create_board(self, board_id: str, user_id: str, name: str = None, description: str = None):
        """Create a new board"""
        board = Board(board_id=board_id, user_id=user_id, name=name, description=description)
        self.session.add(board)
        self.session.commit()
        return board

    def add_pin_to_board(self, pin_id: str, board_id: str):
        """Add a pin to a board"""
        # Check if association already exists
        existing = self.session.query(PinBoard).filter(
            PinBoard.pin_id == pin_id, PinBoard.board_id == board_id
        ).first()
        if not existing:
            pin_board = PinBoard(pin_id=pin_id, board_id=board_id)
            self.session.add(pin_board)
            self.session.commit()
        return existing or pin_board

    def create_topic(self, topic_id: str, name: str, description: str = None, topic_vector: np.ndarray = None):
        """Create a new topic"""
        topic = Topic(
            topic_id=topic_id, 
            name=name, 
            description=description,
            topic_vector=topic_vector.tolist() if topic_vector is not None else None
        )
        self.session.add(topic)
        self.session.commit()
        return topic

    def add_pin_to_topic(self, pin_id: str, topic_id: str, confidence: float = 1.0):
        """Associate a pin with a topic"""
        existing = self.session.query(PinTopic).filter(
            PinTopic.pin_id == pin_id, PinTopic.topic_id == topic_id
        ).first()
        if existing:
            existing.confidence = confidence
        else:
            pin_topic = PinTopic(pin_id=pin_id, topic_id=topic_id, confidence=confidence)
            self.session.add(pin_topic)
        self.session.commit()
        return existing or pin_topic

    def update_user_topic_affinity(self, user_id: str, topic_id: str, affinity_delta: float, interaction_type: str = None):
        """Update user's affinity to a topic based on interaction"""
        user_topic = self.session.query(UserTopic).filter(
            UserTopic.user_id == user_id, UserTopic.topic_id == topic_id
        ).first()
        
        if not user_topic:
            user_topic = UserTopic(user_id=user_id, topic_id=topic_id)
            self.session.add(user_topic)
        
        user_topic.affinity_score += affinity_delta
        user_topic.interaction_count += 1
        user_topic.last_interaction = datetime.utcnow()
        user_topic.updated_at = datetime.utcnow()
        self.session.commit()
        return user_topic

    def add_taste_graph_edge(self, source_type: str, source_id: str, target_type: str, target_id: str, 
                           edge_type: str, weight: float = 1.0, metadata: dict = None):
        """Add or update an edge in the taste graph"""
        edge = self.session.query(TasteGraphEdge).filter(
            TasteGraphEdge.source_type == source_type,
            TasteGraphEdge.source_id == source_id,
            TasteGraphEdge.target_type == target_type,
            TasteGraphEdge.target_id == target_id,
            TasteGraphEdge.edge_type == edge_type
        ).first()
        
        if edge:
            edge.weight = weight
            edge.edge_data = metadata
            edge.updated_at = datetime.utcnow()
        else:
            edge = TasteGraphEdge(
                source_type=source_type, source_id=source_id,
                target_type=target_type, target_id=target_id,
                edge_type=edge_type, weight=weight, edge_data=metadata
            )
            self.session.add(edge)
        self.session.commit()
        return edge

    def get_user_topics(self, user_id: str, top_k: int = 20):
        """Get user's top topics by affinity"""
        topics = self.session.query(UserTopic).filter(
            UserTopic.user_id == user_id
        ).order_by(UserTopic.affinity_score.desc()).limit(top_k).all()
        return [(t.topic_id, t.affinity_score) for t in topics]

    def get_topic_pins(self, topic_id: str, top_k: int = 50):
        """Get pins associated with a topic"""
        pin_topics = self.session.query(PinTopic).filter(
            PinTopic.topic_id == topic_id
        ).order_by(PinTopic.confidence.desc()).limit(top_k).all()
        return [(pt.pin_id, pt.confidence) for pt in pin_topics]

    def get_user_interaction_history(self, user_id: str, limit: int = 1000):
        """Get user's interaction history for taste graph analysis"""
        interactions = self.session.query(Interaction).filter(
            Interaction.user_id == user_id
        ).order_by(Interaction.created_at.desc()).limit(limit).all()
        return interactions

def get_session():
    return DBSession()
