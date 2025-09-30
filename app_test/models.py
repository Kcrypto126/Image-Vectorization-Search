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
