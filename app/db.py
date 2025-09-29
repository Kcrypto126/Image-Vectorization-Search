# app/db.py
from sqlalchemy import create_engine, Column, String, DateTime, JSON, inspect, text, Integer, Float, Index
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os
import uuid
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
    # 'metadata' is reserved by SQLAlchemy's Declarative API; map to column name 'metadata'
    extra_metadata = Column('metadata', JSON, nullable=True)
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
            extra_metadata=metadata,
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

def get_session():
    return DBSession()
