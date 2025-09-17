# app/db.py
from sqlalchemy import create_engine, Column, String, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:Roastery818@localhost:5432/imagedb1")

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
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db(database_url: str = None):
    if database_url:
        global engine, SessionLocal
        engine = create_engine(database_url, pool_pre_ping=True)
        SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    Base.metadata.create_all(engine)

class DBSession:
    def __init__(self):
        self.session = SessionLocal()

    def add_image(self, id_, content_type, image_url, metadata=None):
        img = ImageMetadata(id=id_, content_type=content_type, image_url=image_url, metadata=metadata)
        self.session.add(img)
        self.session.commit()

    def get_image_metadata(self, id_):
        return self.session.get(ImageMetadata, id_)

def get_session():
    return DBSession()
