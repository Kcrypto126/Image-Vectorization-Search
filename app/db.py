# app/db.py
from sqlalchemy import create_engine, Column, String, DateTime, JSON, inspect, text, Integer, Float, Index, Text, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import IntegrityError
from datetime import datetime
import os
import uuid
import numpy as np
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

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
    board_type = Column(String, nullable=False, default='user')  # 'user' or 'auto'
    is_active = Column(Boolean, default=True)  # For auto-boards that can be merged/split
    cluster_center = Column(JSON, nullable=True)  # For auto-boards: original cluster center
    cluster_radius = Column(Float, nullable=True)  # For auto-boards: cluster radius
    pin_count = Column(Integer, default=0)  # Cached pin count for performance
    last_activity = Column(DateTime, default=datetime.utcnow)  # Last time board was updated
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

class UserCorrelation(Base):
    __tablename__ = "user_correlations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_a = Column(String, nullable=False, index=True)
    user_b = Column(String, nullable=False, index=True)
    similarity_score = Column(Float, nullable=False)  # Taste similarity between users
    interaction_overlap = Column(Float, default=0.0)  # Overlap in interacted content
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

Index('ix_user_correlation', UserCorrelation.user_a, UserCorrelation.user_b, unique=True)

class PinCorrelation(Base):
    __tablename__ = "pin_correlations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    pin_a = Column(String, nullable=False, index=True)
    pin_b = Column(String, nullable=False, index=True)
    similarity_score = Column(Float, nullable=False)  # Vector similarity
    cooccurrence_score = Column(Float, default=0.0)  # How often they appear together
    behavioral_score = Column(Float, default=0.0)  # How often users interact with both
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

Index('ix_pin_correlation', PinCorrelation.pin_a, PinCorrelation.pin_b, unique=True)

class BoardEvolution(Base):
    __tablename__ = "board_evolution"
    id = Column(Integer, primary_key=True, autoincrement=True)
    board_id = Column(String, nullable=False, index=True)
    evolution_type = Column(String, nullable=False)  # 'created', 'merged', 'split', 'updated'
    previous_board_ids = Column(JSON, nullable=True)  # For merge/split operations
    new_board_ids = Column(JSON, nullable=True)  # For split operations
    reason = Column(Text, nullable=True)  # Human-readable reason
    evolution_data = Column(JSON, nullable=True)  # Additional evolution data (renamed from metadata)
    created_at = Column(DateTime, default=datetime.utcnow)

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
        
        # Migrate boards table for new auto-board features
        try:
            board_cols = {c['name'] for c in inspector.get_columns('boards')}
            with engine.begin() as conn:
                # Add new columns for auto-board functionality
                if 'board_type' not in board_cols:
                    try:
                        conn.execute(text("ALTER TABLE boards ADD COLUMN board_type VARCHAR DEFAULT 'user'"))
                    except Exception:
                        pass
                
                if 'is_active' not in board_cols:
                    try:
                        conn.execute(text("ALTER TABLE boards ADD COLUMN is_active BOOLEAN DEFAULT true"))
                    except Exception:
                        pass
                
                if 'cluster_center' not in board_cols:
                    try:
                        conn.execute(text("ALTER TABLE boards ADD COLUMN cluster_center JSON"))
                    except Exception:
                        pass
                
                if 'cluster_radius' not in board_cols:
                    try:
                        conn.execute(text("ALTER TABLE boards ADD COLUMN cluster_radius FLOAT"))
                    except Exception:
                        pass
                
                if 'pin_count' not in board_cols:
                    try:
                        conn.execute(text("ALTER TABLE boards ADD COLUMN pin_count INTEGER DEFAULT 0"))
                    except Exception:
                        pass
                
                if 'last_activity' not in board_cols:
                    try:
                        conn.execute(text("ALTER TABLE boards ADD COLUMN last_activity TIMESTAMP"))
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Failed to migrate boards table: {e}")
        
        # Create new tables for correlations and board evolution
        try:
            # Check if new tables exist, if not create them
            existing_tables = inspector.get_table_names()
            
            with engine.begin() as conn:
                if 'user_correlations' not in existing_tables:
                    try:
                        conn.execute(text("""
                            CREATE TABLE user_correlations (
                                id SERIAL PRIMARY KEY,
                                user_a VARCHAR NOT NULL,
                                user_b VARCHAR NOT NULL,
                                similarity_score FLOAT NOT NULL,
                                interaction_overlap FLOAT DEFAULT 0.0,
                                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            )
                        """))
                        conn.execute(text("CREATE INDEX ix_user_correlation ON user_correlations (user_a, user_b)"))
                    except Exception as e:
                        logger.warning(f"Failed to create user_correlations table: {e}")
                
                if 'pin_correlations' not in existing_tables:
                    try:
                        conn.execute(text("""
                            CREATE TABLE pin_correlations (
                                id SERIAL PRIMARY KEY,
                                pin_a VARCHAR NOT NULL,
                                pin_b VARCHAR NOT NULL,
                                similarity_score FLOAT NOT NULL,
                                cooccurrence_score FLOAT DEFAULT 0.0,
                                behavioral_score FLOAT DEFAULT 0.0,
                                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            )
                        """))
                        conn.execute(text("CREATE INDEX ix_pin_correlation ON pin_correlations (pin_a, pin_b)"))
                    except Exception as e:
                        logger.warning(f"Failed to create pin_correlations table: {e}")
                
                if 'board_evolution' not in existing_tables:
                    try:
                        conn.execute(text("""
                            CREATE TABLE board_evolution (
                                id SERIAL PRIMARY KEY,
                                board_id VARCHAR NOT NULL,
                                evolution_type VARCHAR NOT NULL,
                                previous_board_ids JSON,
                                new_board_ids JSON,
                                reason TEXT,
                                evolution_data JSON,
                                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            )
                        """))
                        conn.execute(text("CREATE INDEX ix_board_evolution ON board_evolution (board_id)"))
                    except Exception as e:
                        logger.warning(f"Failed to create board_evolution table: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to create new tables: {e}")
            
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
    
    def get_all_images(self):
        """Get all images from the database"""
        return self.session.query(ImageMetadata).all()

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

    def create_board(self, board_id: str, user_id: str, name: str = None, description: str = None, 
                    board_type: str = 'user', board_vector: np.ndarray = None):
        """Create a new board with automatic vector initialization"""
        # If the board already exists, return it instead of failing
        existing = self.session.query(Board).filter(Board.board_id == board_id).first()
        if existing:
            return existing
        
        # Initialize board vector based on type
        if board_vector is not None:
            # Use provided vector (for auto-boards)
            initial_vector = board_vector.tolist()
        elif board_type == 'user':
            # Initialize user boards with zero vector
            initial_vector = [0.0] * 512  # Assuming 512-dimensional vectors
        else:
            # Auto boards should always have a proper vector
            initial_vector = None
        
        board = Board(
            board_id=board_id, 
            user_id=user_id, 
            name=name, 
            description=description,
            board_type=board_type,
            board_vector=initial_vector,
            pin_count=0,
            last_activity=datetime.utcnow()
        )
        try:
            self.session.add(board)
            self.session.commit()
        except IntegrityError:
            # Rollback the failed transaction and return the existing board
            self.session.rollback()
            board = self.session.query(Board).filter(Board.board_id == board_id).first()
        return board
    
    def create_auto_board(self, user_id: str, name: str, description: str, 
                         board_vector: np.ndarray, cluster_center: np.ndarray, 
                         cluster_radius: float = None) -> str:
        """Create an auto-generated board"""
        board_id = f"auto_{user_id}_{uuid.uuid4().hex[:8]}"
        board = Board(
            board_id=board_id,
            user_id=user_id,
            name=name,
            description=description,
            board_type='auto',
            board_vector=board_vector.tolist(),
            cluster_center=cluster_center.tolist(),
            cluster_radius=cluster_radius
        )
        self.session.add(board)
        self.session.commit()
        return board_id
    
    def update_board_vector(self, board_id: str, board_vector: np.ndarray):
        """Update board's vector representation"""
        board = self.session.query(Board).filter(Board.board_id == board_id).first()
        if board:
            board.board_vector = board_vector.tolist()
            board.last_activity = datetime.utcnow()
            self.session.commit()
        return board
    
    def get_user_boards(self, user_id: str, board_type: str = None, active_only: bool = True):
        """Get user's boards, optionally filtered by type"""
        query = self.session.query(Board).filter(Board.user_id == user_id)
        if board_type:
            query = query.filter(Board.board_type == board_type)
        if active_only:
            query = query.filter(Board.is_active == True)
        return query.all()
    
    def get_board_pins(self, board_id: str):
        """Get all pins in a board"""
        return self.session.query(PinBoard).filter(PinBoard.board_id == board_id).all()
    
    def get_board_vector(self, board_id: str):
        """Get board's vector representation"""
        board = self.session.query(Board).filter(Board.board_id == board_id).first()
        if board and board.board_vector:
            return np.array(board.board_vector)
        return None

    def add_pin_to_board(self, pin_id: str, board_id: str, owner_user_id: str = None, 
                        vectorizer=None, index_adapter=None):
        """Add a pin to a board and update board vector/metadata automatically."""
        # Ownership check if requested
        if owner_user_id is not None:
            board = self.session.query(Board).filter(Board.board_id == board_id).first()
            if not board or board.user_id != owner_user_id:
                raise PermissionError("User does not own this board")
        
        # Check if association already exists
        existing = self.session.query(PinBoard).filter(
            PinBoard.pin_id == pin_id, PinBoard.board_id == board_id
        ).first()
        
        if not existing:
            pin_board = PinBoard(pin_id=pin_id, board_id=board_id)
            try:
                self.session.add(pin_board)
                self.session.commit()
                
                # Update board metadata and vector after successful pin addition
                self._update_board_metadata_and_vector(board_id, vectorizer, index_adapter)
                
            except IntegrityError:
                self.session.rollback()
                existing = self.session.query(PinBoard).filter(
                    PinBoard.pin_id == pin_id, PinBoard.board_id == board_id
                ).first()
        else:
            # Pin already exists, still update metadata (in case of external changes)
            self._update_board_metadata_and_vector(board_id, vectorizer, index_adapter)
            
        return existing or pin_board

    def remove_pin_from_board(self, pin_id: str, board_id: str, owner_user_id: str = None,
                             vectorizer=None, index_adapter=None):
        """Remove a pin from a board and update board vector/metadata automatically."""
        # Ownership check if requested
        if owner_user_id is not None:
            board = self.session.query(Board).filter(Board.board_id == board_id).first()
            if not board or board.user_id != owner_user_id:
                raise PermissionError("User does not own this board")
        
        # Remove the association
        pin_board = self.session.query(PinBoard).filter(
            PinBoard.pin_id == pin_id, PinBoard.board_id == board_id
        ).first()
        
        if pin_board:
            self.session.delete(pin_board)
            self.session.commit()
            
            # Update board metadata and vector after successful pin removal
            self._update_board_metadata_and_vector(board_id, vectorizer, index_adapter)
            
        return pin_board

    def _update_board_metadata_and_vector(self, board_id: str, vectorizer=None, index_adapter=None):
        """Update board metadata (pin_count, last_activity) and vector based on pins"""
        try:
            # Get board
            board = self.session.query(Board).filter(Board.board_id == board_id).first()
            if not board:
                return
            
            # Update metadata
            pin_count = self.session.query(PinBoard).filter(PinBoard.board_id == board_id).count()
            board.pin_count = pin_count
            board.last_activity = datetime.utcnow()
            
            # Update vector based on board type
            if board.board_type == 'user' and pin_count > 0 and vectorizer and index_adapter:
                # For user boards, compute vector from all pins
                pin_ids = [pb.pin_id for pb in self.session.query(PinBoard).filter(PinBoard.board_id == board_id).all()]
                if pin_ids:
                    vectors = []
                    for pin_id in pin_ids:
                        try:
                            vector = vectorizer.get_vector(pin_id, index_adapter)
                            if vector is not None:
                                vectors.append(vector)
                        except Exception as e:
                            logger.warning(f"Failed to get vector for pin {pin_id}: {e}")
                    
                    if vectors:
                        # Compute average vector
                        board_vector = np.mean(vectors, axis=0)
                        board.board_vector = board_vector.tolist()
                    else:
                        # Fallback to zero vector if no vectors found
                        board.board_vector = [0.0] * 512
                else:
                    # No pins, keep zero vector
                    board.board_vector = [0.0] * 512
            
            self.session.commit()
            
        except Exception as e:
            logger.warning(f"Failed to update board metadata/vector for {board_id}: {e}")
            self.session.rollback()

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

    def update_user_topic_affinity(self, user_id: str, topic_id: str, affinity_delta: float, interaction_type: str = None, vectorizer=None, index_adapter=None):
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

        # Ensure topic vector exists/updated when user-topic is created/updated
        try:
            if vectorizer is not None:
                # Update topic vector opportunistically every 10 interactions or on first creation
                if user_topic.interaction_count == 1 or (user_topic.interaction_count % 10 == 0):
                    self.ensure_topic_vector(topic_id, vectorizer, index_adapter)
        except Exception as e:
            logger.warning(f"Failed to ensure topic vector for {topic_id}: {e}")
        return user_topic

    def ensure_topic_vector(self, topic_id: str, vectorizer, index_adapter=None):
        """Ensure Topic.topic_vector is generated/updated from associated pins or topic name."""
        topic = self.session.query(Topic).filter(Topic.topic_id == topic_id).first()
        if not topic:
            return None
        try:
            # Try to aggregate vectors from associated pins
            pin_topics = self.session.query(PinTopic).filter(PinTopic.topic_id == topic_id).all()
            vectors = []
            if pin_topics and index_adapter is not None:
                for pt in pin_topics:
                    try:
                        vec = vectorizer.get_vector(pt.pin_id, index_adapter)
                        if vec is not None:
                            vectors.append(vec)
                    except Exception as e:
                        logger.warning(f"Failed to fetch vector for pin {pt.pin_id}: {e}")
            if vectors:
                topic_vec = np.mean(vectors, axis=0)
            else:
                # Fallback to text encoding of topic name
                topic_vec = vectorizer.text_to_vector(topic.name)[0]
            # Normalize and persist
            topic_vec = topic_vec / (np.linalg.norm(topic_vec) + 1e-8)
            topic.topic_vector = topic_vec.tolist()
            topic.updated_at = datetime.utcnow()
            self.session.commit()
            return topic_vec
        except Exception as e:
            logger.warning(f"ensure_topic_vector failed for {topic_id}: {e}")
            self.session.rollback()
            return None

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
    
    # User correlation methods
    def update_user_correlation(self, user_a: str, user_b: str, similarity_score: float, 
                               interaction_overlap: float = 0.0):
        """Update correlation between two users"""
        correlation = self.session.query(UserCorrelation).filter(
            ((UserCorrelation.user_a == user_a) & (UserCorrelation.user_b == user_b)) |
            ((UserCorrelation.user_a == user_b) & (UserCorrelation.user_b == user_a))
        ).first()
        
        if correlation:
            correlation.similarity_score = similarity_score
            correlation.interaction_overlap = interaction_overlap
            correlation.last_updated = datetime.utcnow()
        else:
            correlation = UserCorrelation(
                user_a=user_a, user_b=user_b, 
                similarity_score=similarity_score, 
                interaction_overlap=interaction_overlap
            )
            self.session.add(correlation)
        self.session.commit()
        return correlation
    
    def get_similar_users(self, user_id: str, top_k: int = 20):
        """Get users similar to the given user"""
        correlations = self.session.query(UserCorrelation).filter(
            (UserCorrelation.user_a == user_id) | (UserCorrelation.user_b == user_id)
        ).order_by(UserCorrelation.similarity_score.desc()).limit(top_k).all()
        
        similar_users = []
        for corr in correlations:
            other_user = corr.user_b if corr.user_a == user_id else corr.user_a
            similar_users.append((other_user, corr.similarity_score, corr.interaction_overlap))
        return similar_users
    
    # Pin correlation methods
    def update_pin_correlation(self, pin_a: str, pin_b: str, similarity_score: float,
                              cooccurrence_score: float = 0.0, behavioral_score: float = 0.0):
        """Update correlation between two pins"""
        correlation = self.session.query(PinCorrelation).filter(
            ((PinCorrelation.pin_a == pin_a) & (PinCorrelation.pin_b == pin_b)) |
            ((PinCorrelation.pin_a == pin_b) & (PinCorrelation.pin_b == pin_a))
        ).first()
        
        if correlation:
            correlation.similarity_score = similarity_score
            correlation.cooccurrence_score = cooccurrence_score
            correlation.behavioral_score = behavioral_score
            correlation.last_updated = datetime.utcnow()
        else:
            correlation = PinCorrelation(
                pin_a=pin_a, pin_b=pin_b,
                similarity_score=similarity_score,
                cooccurrence_score=cooccurrence_score,
                behavioral_score=behavioral_score
            )
            self.session.add(correlation)
        self.session.commit()
        return correlation
    
    def get_similar_pins(self, pin_id: str, top_k: int = 50):
        """Get pins similar to the given pin"""
        correlations = self.session.query(PinCorrelation).filter(
            (PinCorrelation.pin_a == pin_id) | (PinCorrelation.pin_b == pin_id)
        ).order_by(PinCorrelation.similarity_score.desc()).limit(top_k).all()
        
        similar_pins = []
        for corr in correlations:
            other_pin = corr.pin_b if corr.pin_a == pin_id else corr.pin_a
            similar_pins.append((other_pin, corr.similarity_score, corr.cooccurrence_score, corr.behavioral_score))
        return similar_pins
    
    # Board evolution tracking
    def track_board_evolution(self, board_id: str, evolution_type: str, 
                             previous_board_ids: list = None, new_board_ids: list = None,
                             reason: str = None, evolution_data: dict = None):
        """Track board evolution events"""
        evolution = BoardEvolution(
            board_id=board_id,
            evolution_type=evolution_type,
            previous_board_ids=previous_board_ids,
            new_board_ids=new_board_ids,
            reason=reason,
            evolution_data=evolution_data
        )
        self.session.add(evolution)
        self.session.commit()
        return evolution

def get_session():
    return DBSession()

def force_migration():
    """
    Force database migration to add new columns and tables.
    This can be called manually if automatic migration fails.
    """
    try:
        inspector = inspect(engine)
        
        # Migrate boards table
        board_cols = {c['name'] for c in inspector.get_columns('boards')}
        with engine.begin() as conn:
            if 'board_type' not in board_cols:
                conn.execute(text("ALTER TABLE boards ADD COLUMN board_type VARCHAR DEFAULT 'user'"))
            if 'is_active' not in board_cols:
                conn.execute(text("ALTER TABLE boards ADD COLUMN is_active BOOLEAN DEFAULT true"))
            if 'cluster_center' not in board_cols:
                conn.execute(text("ALTER TABLE boards ADD COLUMN cluster_center JSON"))
            if 'cluster_radius' not in board_cols:
                conn.execute(text("ALTER TABLE boards ADD COLUMN cluster_radius FLOAT"))
            if 'pin_count' not in board_cols:
                conn.execute(text("ALTER TABLE boards ADD COLUMN pin_count INTEGER DEFAULT 0"))
            if 'last_activity' not in board_cols:
                conn.execute(text("ALTER TABLE boards ADD COLUMN last_activity TIMESTAMP"))
        
        # Create new tables
        existing_tables = inspector.get_table_names()
        with engine.begin() as conn:
            if 'user_correlations' not in existing_tables:
                conn.execute(text("""
                    CREATE TABLE user_correlations (
                        id SERIAL PRIMARY KEY,
                        user_a VARCHAR NOT NULL,
                        user_b VARCHAR NOT NULL,
                        similarity_score FLOAT NOT NULL,
                        interaction_overlap FLOAT DEFAULT 0.0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                conn.execute(text("CREATE INDEX ix_user_correlation ON user_correlations (user_a, user_b)"))
            
            if 'pin_correlations' not in existing_tables:
                conn.execute(text("""
                    CREATE TABLE pin_correlations (
                        id SERIAL PRIMARY KEY,
                        pin_a VARCHAR NOT NULL,
                        pin_b VARCHAR NOT NULL,
                        similarity_score FLOAT NOT NULL,
                        cooccurrence_score FLOAT DEFAULT 0.0,
                        behavioral_score FLOAT DEFAULT 0.0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                conn.execute(text("CREATE INDEX ix_pin_correlation ON pin_correlations (pin_a, pin_b)"))
            
            if 'board_evolution' not in existing_tables:
                conn.execute(text("""
                    CREATE TABLE board_evolution (
                        id SERIAL PRIMARY KEY,
                        board_id VARCHAR NOT NULL,
                        evolution_type VARCHAR NOT NULL,
                        previous_board_ids JSON,
                        new_board_ids JSON,
                        reason TEXT,
                        evolution_data JSON,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                conn.execute(text("CREATE INDEX ix_board_evolution ON board_evolution (board_id)"))
        
        logger.info("Database migration completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database migration failed: {e}")
        return False
