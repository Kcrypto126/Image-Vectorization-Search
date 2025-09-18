from sqlalchemy import Column, String, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import uuid

Base = declarative_base()

class ImageMetadata(Base):
    __tablename__ = "images"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    content_type = Column(String)
    image_url = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db(uri="postgresql://postgres:Roastery818@localhost:5432/imagedb"):
    engine = create_engine(uri)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)

# def init_db(uri=None):
#     # Prefer explicit uri, then DATABASE_URL env var, default to SQLite for local dev
#     database_url = uri or os.getenv("DATABASE_URL", "sqlite:///./app.db")
#     connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
#     engine = create_engine(database_url, connect_args=connect_args)
#     Base.metadata.create_all(engine)
#     return sessionmaker(bind=engine)
