# app/taste_graph.py
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json

from app.db import Board, PinTopic, Topic, User, Interaction
from app.enhanced_auto_board_manager import EnhancedAutoBoardManager

logger = logging.getLogger(__name__)

class TasteGraphManager:
    """
    High-level manager for the Pinterest-style Taste Graph system.
    Coordinates between PinSage training, real-time updates, and search personalization.
    """
    
    def __init__(self, session, vectorizer, pinsage_system=None, index_adapter=None):
        self.session = session
        self.vectorizer = vectorizer
        self.pinsage_system = pinsage_system
        self.index_adapter = index_adapter
        self.topic_cache = {}
        self.user_cache = {}
        
        # Initialize auto-board creator and correlation engine
        from .auto_board_creator import AutoBoardCreator, BoardVectorManager
        from .correlation_engine import CorrelationEngine
        
        self.auto_board_creator = AutoBoardCreator(session, vectorizer, index_adapter)
        self.board_vector_manager = BoardVectorManager(session, vectorizer, index_adapter)
        self.correlation_engine = CorrelationEngine(session, vectorizer, index_adapter)
        
        # Initialize enhanced auto-board manager
        self.enhanced_auto_board_manager = EnhancedAutoBoardManager(
            session, vectorizer, index_adapter,
            min_pins_for_new_board=15,
            similarity_threshold=0.7,
            merge_threshold=0.9,
            decay_days=30,
            duplicate_threshold=0.95
        )
        
    def initialize_user(self, user_id: str) -> Dict:
        """
        Initialize a new user in the taste graph system.
        Creates user record, initializes taste vector, and creates initial auto-boards.
        """
        logger.info(f"Initializing user {user_id}")
        
        # Create user record
        user = self.session.get_or_create_user(user_id)
        
        # Initialize taste vector with random values
        initial_taste = np.random.normal(0, 0.1, 512)
        initial_taste = initial_taste / (np.linalg.norm(initial_taste) + 1e-8)
        
        self.session.update_user_taste_vector(user_id, initial_taste)
        
        # Cache user
        self.user_cache[user_id] = {
            'taste_vector': initial_taste,
            'last_updated': datetime.utcnow(),
            'interaction_count': 0
        }
        
        # Automatically create initial auto-boards after user has some interactions
        # This will be triggered by the periodic background task
        logger.info(f"User {user_id} initialized - auto-boards will be created after sufficient interactions")
        
        return {
            'user_id': user_id,
            'taste_vector_version': 1,
            'status': 'initialized'
        }
    
    def create_auto_boards_for_user(self, user_id: str) -> Dict:
        """Create auto-generated boards for a user"""
        try:
            created_boards = self.auto_board_creator.create_auto_boards_for_user(user_id)
            return {
                'user_id': user_id,
                'created_boards': created_boards,
                'count': len(created_boards),
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Failed to create auto-boards for user {user_id}: {e}")
            return {
                'user_id': user_id,
                'created_boards': [],
                'count': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def update_auto_boards_for_user(self, user_id: str) -> Dict:
        """Update existing auto-boards for a user"""
        try:
            updated_boards = self.auto_board_creator.update_auto_boards(user_id)
            return {
                'user_id': user_id,
                'updated_boards': updated_boards,
                'count': len(updated_boards),
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Failed to update auto-boards for user {user_id}: {e}")
            return {
                'user_id': user_id,
                'updated_boards': [],
                'count': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def compute_correlations_for_user(self, user_id: str) -> Dict:
        """Compute correlations for a user"""
        try:
            user_correlations = self.correlation_engine.compute_user_correlations(user_id)
            return {
                'user_id': user_id,
                'correlations_users_count': len(user_correlations),
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Failed to compute correlations for user {user_id}: {e}")
            return {
                'user_id': user_id,
                'correlations_count': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def get_recommendations_for_user(self, user_id: str, top_k: int = 20) -> List[Dict]:
        """Get personalized recommendations for a user"""
        try:
            recommendations = self.correlation_engine.get_recommendations_for_user(user_id, top_k)
            return [
                {
                    'pin_id': pin_id,
                    'score': score,
                    'reason': reason
                }
                for pin_id, score, reason in recommendations
            ]
        except Exception as e:
            logger.error(f"Failed to get recommendations for user {user_id}: {e}")
            return []
    
    def process_user_interaction(self, user_id: str, image_id: str, 
                               event: str, weight: float = 1.0) -> Dict:
        """
        Process a user interaction and update taste graph accordingly.
        """
        logger.info(f"Processing {event} interaction for user {user_id} with image {image_id}")
        
        # Record interaction in database
        self.session.record_interaction(user_id, image_id, event, weight)
        
        # Real-time taste vector update
        if self.pinsage_system and self.pinsage_system.get('realtime_updater'):
            self.pinsage_system['realtime_updater'].update_user_taste_on_interaction(
                user_id, image_id, event, weight
            )
        
        # Update user cache
        if user_id in self.user_cache:
            self.user_cache[user_id]['interaction_count'] += 1
            self.user_cache[user_id]['last_updated'] = datetime.utcnow()
        
        # Extract and update topics
        self._update_image_topics(image_id)
        self._update_user_topic_affinities(user_id, image_id, event, weight)
        
        # Process enhanced auto-board logic
        try:
            self.enhanced_auto_board_manager.process_user_interaction(user_id, image_id, event, weight)
        except Exception as e:
            logger.warning(f"Enhanced auto-board processing failed: {e}")
        
        # Update correlations (in background)
        try:
            self.correlation_engine.compute_user_correlations(user_id, max_users=20)
            self.correlation_engine.compute_pin_correlations(image_id, max_pins=50)
        except Exception as e:
            logger.warning(f"Correlation update failed: {e}")
        
        return {
            'user_id': user_id,
            'image_id': image_id,
            'event': event,
            'weight': weight,
            'status': 'processed'
        }
    
    def _update_image_topics(self, image_id: str):
        """
        Extract topics from image metadata and create topic associations.
        """
        # Get image metadata
        image_meta = self.session.get_image_metadata(image_id)
        if not image_meta:
            return
        
        # Extract topics from metadata
        topics = self._extract_topics_from_metadata(image_meta)
        
        # Create topic associations
        for topic_name, confidence in topics:
            topic_id = self._get_or_create_topic(topic_name)
            self.session.add_pin_to_topic(image_id, topic_id, confidence)
    
    def _extract_topics_from_metadata(self, image_meta) -> List[Tuple[str, float]]:
        """
        Extract topic names and confidence scores from image metadata.
        """
        topics = []
        
        # Extract from generated metadata
        if image_meta.generated_metadata:
            gen_meta = image_meta.generated_metadata
            
            # Categories as topics
            categories = gen_meta.get('Categories', [])
            for cat in categories[:3]:  # Top 3 categories
                topics.append((cat.lower(), 0.8))
            
            # Visual style as topics
            visual_styles = gen_meta.get('Visual style', [])
            for style in visual_styles[:2]:  # Top 2 styles
                topics.append((style.lower(), 0.6))
            
            # Keywords as topics
            keywords = gen_meta.get('Keywords', [])
            for keyword in keywords[:5]:  # Top 5 keywords
                topics.append((keyword.lower(), 0.4))
        
        # Extract from raw detections
        if image_meta.raw_detections:
            raw_meta = image_meta.raw_detections
            
            # Objects as topics
            objects = raw_meta.get('objects', [])
            for obj in objects[:3]:  # Top 3 objects
                topics.append((obj.lower(), 0.7))
            
            # Colors as topics
            colors = raw_meta.get('colors_named', [])
            for color in colors[:2]:  # Top 2 colors
                topics.append((color.lower(), 0.5))
        
        return topics
    
    def _get_or_create_topic(self, topic_name: str) -> str:
        """
        Get existing topic or create new one.
        """
        # Check cache first
        if topic_name in self.topic_cache:
            return self.topic_cache[topic_name]
        
        # Check database
        existing_topic = self.session.session.query(Topic).filter(
            Topic.topic_id == topic_name
        ).first()
        
        if existing_topic:
            self.topic_cache[topic_name] = existing_topic.topic_id
            return existing_topic.topic_id
        
        # Create new topic
        topic = self.session.create_topic(
            topic_id=topic_name,
            name=topic_name.title(),
            description=f"Topic: {topic_name}"
        )
        # Generate initial topic_vector automatically
        try:
            self.session.ensure_topic_vector(topic.topic_id, self.vectorizer, self.index_adapter)
        except Exception as e:
            logger.warning(f"Failed to initialize topic vector for {topic_name}: {e}")
        
        self.topic_cache[topic_name] = topic.topic_id
        return topic.topic_id
    
    def _update_user_topic_affinities(self, user_id: str, image_id: str, 
                                    event: str, weight: float):
        """
        Update user's topic affinities based on interaction.
        """
        # Get topics associated with this image
        pin_topics = self.session.session.query(PinTopic).filter(
            PinTopic.pin_id == image_id
        ).all()
        
        # Calculate affinity delta based on event type
        event_affinities = {
            'save': 0.3,
            'like': 0.2,
            'click': 0.1,
            'view': 0.05
        }
        
        affinity_delta = event_affinities.get(event, 0.1) * weight
        
        # Update affinity for each topic
        for pin_topic in pin_topics:
            self.session.update_user_topic_affinity(
                user_id, pin_topic.topic_id, affinity_delta, event
            )
    
    def get_personalized_search_results(self, user_id: str, search_results: List[Dict], 
                                      top_k: int = 10) -> List[Dict]:
        """
        Rerank search results using user's taste vector, board vectors, and enhanced correlations.
        """
        if not user_id:
            return search_results[:top_k]
        
        # Get user taste vector
        user_taste = self.session.get_user_taste_vector(user_id)
        if user_taste is None:
            return search_results[:top_k]
        
        # Get user's boards and their vectors
        user_boards = self.session.get_user_boards(user_id, active_only=True)
        board_vectors = {}
        for board in user_boards:
            if board.board_vector:
                board_vectors[board.board_id] = np.array(board.board_vector)
        
        # Get user's top topics
        user_topics = dict(self.session.get_user_topics(user_id, 20))
        
        # Get similar users for collaborative filtering
        similar_users = self.session.get_similar_users(user_id, top_k=10)
        similar_user_ids = [user_id for user_id, _, _ in similar_users]
        
        # Rerank results
        reranked_results = []
        
        for result in search_results:
            image_id = result['id']
            base_score = result['score']
            
            # Calculate taste similarity bonus
            taste_bonus = self._calculate_taste_similarity_bonus(user_taste, image_id)
            
            # Calculate board similarity bonus
            board_bonus = self._calculate_board_similarity_bonus(board_vectors, image_id)
            
            # Calculate topic affinity bonus
            topic_bonus = self._calculate_topic_affinity_bonus(user_topics, image_id)
            
            # Calculate collaborative filtering bonus
            collaborative_bonus = self._calculate_collaborative_bonus(similar_user_ids, image_id)
            
            # Calculate behavioral correlation bonus
            behavioral_bonus = self._calculate_behavioral_bonus(user_id, image_id)
            
            # Combine scores with weights
            personalized_score = (base_score + 
                                taste_bonus * 0.3 + 
                                board_bonus * 0.25 + 
                                topic_bonus * 0.2 + 
                                collaborative_bonus * 0.15 + 
                                behavioral_bonus * 0.1)
            
            reranked_results.append({
                **result,
                'score': personalized_score,
                'taste_bonus': taste_bonus,
                'board_bonus': board_bonus,
                'topic_bonus': topic_bonus,
                'collaborative_bonus': collaborative_bonus,
                'behavioral_bonus': behavioral_bonus
            })
        
        # Sort by personalized score
        reranked_results.sort(key=lambda x: x['score'], reverse=True)
        
        return reranked_results[:top_k]
    
    def _calculate_taste_similarity_bonus(self, user_taste: np.ndarray, image_id: str) -> float:
        """
        Calculate bonus score based on taste vector similarity.
        """
        # Get exact image vector from index if available
        image_vector = None
        if self.index_adapter:
            try:
                image_vector = self.vectorizer.get_vector(image_id, self.index_adapter)
            except Exception as e:
                logger.warning(f"Failed to get vector for image {image_id}: {e}")
        
        if image_vector is None:
            # Fallback to random vector if not found in index
            image_vector = np.random.normal(0, 0.1, 512)
            image_vector = image_vector / (np.linalg.norm(image_vector) + 1e-8)
        
        # Calculate cosine similarity
        similarity = np.dot(user_taste, image_vector)
        
        # Convert to bonus score (0 to 0.3 range)
        bonus = max(0, similarity * 0.3)
        
        return bonus
    
    def _calculate_board_similarity_bonus(self, board_vectors: Dict[str, np.ndarray], image_id: str) -> float:
        """
        Calculate bonus score based on board vector similarity.
        """
        if not board_vectors:
            return 0.0
        
        # Get exact image vector from index if available
        image_vector = None
        if self.index_adapter:
            try:
                image_vector = self.vectorizer.get_vector(image_id, self.index_adapter)
            except Exception as e:
                logger.warning(f"Failed to get vector for image {image_id}: {e}")
        
        if image_vector is None:
            return 0.0
        
        # Calculate similarity with each board
        max_similarity = 0.0
        for board_id, board_vector in board_vectors.items():
            similarity = np.dot(image_vector, board_vector)
            max_similarity = max(max_similarity, similarity)
        
        # Convert to bonus score (0 to 0.4 range)
        bonus = max(0, max_similarity * 0.4)
        return bonus
    
    def _calculate_collaborative_bonus(self, similar_user_ids: List[str], image_id: str) -> float:
        """
        Calculate bonus score based on similar users' preferences.
        """
        if not similar_user_ids:
            return 0.0
        
        # Check if any similar users interacted with this image
        interactions = self.session.session.query(Interaction).filter(
            Interaction.image_id == image_id,
            Interaction.user_id.in_(similar_user_ids)
        ).all()
        
        if not interactions:
            return 0.0
        
        # Calculate weighted score based on interaction types
        total_score = 0.0
        for interaction in interactions:
            weight = self._get_interaction_weight(interaction.event, interaction.weight)
            total_score += weight
        
        # Normalize and convert to bonus
        bonus = min(0.3, total_score / len(similar_user_ids) * 0.3)
        return bonus
    
    def _calculate_behavioral_bonus(self, user_id: str, image_id: str) -> float:
        """
        Calculate bonus score based on behavioral correlations.
        """
        # Get similar pins to this one
        similar_pins = self.session.get_similar_pins(image_id, top_k=20)
        
        # Check if user interacted with any similar pins
        user_interactions = self.session.get_user_interaction_history(user_id, limit=1000)
        user_pin_ids = set(interaction.image_id for interaction in user_interactions)
        
        behavioral_score = 0.0
        for similar_pin_id, similarity, cooccurrence, behavioral in similar_pins:
            if similar_pin_id in user_pin_ids:
                # User interacted with similar content
                behavioral_score += behavioral * 0.5 + similarity * 0.3 + cooccurrence * 0.2
        
        # Convert to bonus score
        bonus = min(0.2, behavioral_score * 0.2)
        return bonus
    
    def _get_interaction_weight(self, event: str, base_weight: float) -> float:
        """Convert interaction event to weight"""
        event_weights = {
            'save': 2.0,
            'like': 1.5,
            'click': 1.0,
            'view': 0.5
        }
        return event_weights.get(event, 1.0) * base_weight
    
    def _calculate_topic_affinity_bonus(self, user_topics: Dict[str, float], image_id: str) -> float:
        """
        Calculate bonus score based on topic affinities.
        """
        # Get topics associated with this image
        pin_topics = self.session.session.query(PinTopic).filter(
            PinTopic.pin_id == image_id
        ).all()
        
        bonus = 0.0
        
        for pin_topic in pin_topics:
            if pin_topic.topic_id in user_topics:
                user_affinity = user_topics[pin_topic.topic_id]
                topic_confidence = pin_topic.confidence
                
                # Bonus proportional to user affinity and topic confidence
                bonus += user_affinity * topic_confidence * 0.1
        
        return min(bonus, 0.2)  # Cap at 0.2
    
    def create_board(self, user_id: str, board_id: str, name: str, description: str = None) -> Dict:
        """
        Create a new board for a user with automatic vector initialization.
        """
        board = self.session.create_board(board_id, user_id, name, description, board_type='user')
        
        return {
            'board_id': board.board_id,
            'user_id': board.user_id,
            'name': board.name,
            'description': board.description,
            'status': 'created'
        }
    
    def add_pin_to_board(self, pin_id: str, board_id: str, user_id: str = None) -> Dict:
        """
        Add a pin to a board with automatic vector and metadata updates.
        """
        # Enforce that only the owner can add pins if user_id provided
        pin_board = self.session.add_pin_to_board(
            pin_id, board_id, owner_user_id=user_id, 
            vectorizer=self.vectorizer, index_adapter=self.index_adapter
        )
        
        return {
            'pin_id': pin_id,
            'board_id': board_id,
            'status': 'added'
        }
    
    def remove_pin_from_board(self, pin_id: str, board_id: str, user_id: str = None) -> Dict:
        """
        Remove a pin from a board with automatic vector and metadata updates.
        """
        # Enforce that only the owner can remove pins if user_id provided
        pin_board = self.session.remove_pin_from_board(
            pin_id, board_id, owner_user_id=user_id,
            vectorizer=self.vectorizer, index_adapter=self.index_adapter
        )
        
        if pin_board:
            return {
                'pin_id': pin_id,
                'board_id': board_id,
                'status': 'removed'
            }
        else:
            return {
                'pin_id': pin_id,
                'board_id': board_id,
                'status': 'not_found'
            }
    
    def get_user_taste_profile(self, user_id: str) -> Dict:
        """
        Get comprehensive user taste profile.
        """
        # Get user taste vector
        taste_vector = self.session.get_user_taste_vector(user_id)
        
        # Get top topics
        top_topics = self.session.get_user_topics(user_id, 10)
        
        # Get interaction history
        interactions = self.session.get_user_interaction_history(user_id, 100)
        
        # Get user boards
        boards = self.session.session.query(Board).filter(
            Board.user_id == user_id
        ).all()
        
        return {
            'user_id': user_id,
            'taste_vector': taste_vector.tolist() if taste_vector is not None else None,
            'top_topics': [{'topic_id': tid, 'affinity': aff} for tid, aff in top_topics],
            'interaction_count': len(interactions),
            'boards': [{'board_id': b.board_id, 'name': b.name} for b in boards],
            'last_updated': datetime.utcnow().isoformat()
        }
    
    def run_offline_training(self, user_ids: List[str] = None, epochs: int = 10) -> Dict:
        """
        Run offline PinSage training for long-term taste pattern learning.
        """
        if not self.pinsage_system:
            return {'status': 'error', 'message': 'PinSage system not initialized'}
        
        logger.info("Starting offline PinSage training...")
        
        trainer = self.pinsage_system['trainer']
        
        # Get users to train on
        if user_ids is None:
            users = self.session.session.query(User).all()
            user_ids = [u.user_id for u in users]
        
        logger.info(f"Training on {len(user_ids)} users for {epochs} epochs")
        
        # Train for specified epochs
        for epoch in range(epochs):
            loss = trainer.train_epoch(user_ids, batch_size=16)
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
        
        # Update user taste vectors
        trainer.update_user_taste_vectors(user_ids)
        
        logger.info("Offline training completed")
        
        return {
            'status': 'completed',
            'users_trained': len(user_ids),
            'epochs': epochs,
            'final_loss': loss
        }
    
    def rebuild_taste_graph(self) -> Dict:
        """
        Rebuild the entire taste graph from scratch.
        """
        logger.info("Rebuilding taste graph...")
        
        if not self.pinsage_system:
            return {'status': 'error', 'message': 'PinSage system not initialized'}
        
        # Rebuild graph
        graph_builder = self.pinsage_system['graph_builder']
        graph_builder.build_graph()
        
        logger.info("Taste graph rebuilt successfully")
        
        return {
            'status': 'completed',
            'nodes': graph_builder.graph.number_of_nodes(),
            'edges': graph_builder.graph.number_of_edges()
        }

# Global taste graph manager instance
_taste_graph_manager = None

def get_taste_graph_manager(session=None, vectorizer=None, pinsage_system=None):
    """
    Get or create the global taste graph manager.
    """
    global _taste_graph_manager
    
    if _taste_graph_manager is None:
        _taste_graph_manager = TasteGraphManager(session, vectorizer, pinsage_system)
    
    return _taste_graph_manager

def initialize_taste_graph_system(session, vectorizer, index_adapter=None):
    """
    Initialize the complete taste graph system.
    """
    from .pinsage import initialize_taste_graph
    
    logger.info("Initializing complete taste graph system...")
    
    # Initialize PinSage system
    pinsage_system = initialize_taste_graph(session, vectorizer, index_adapter)
    
    # Initialize taste graph manager
    taste_graph_manager = TasteGraphManager(session, vectorizer, pinsage_system, index_adapter)
    
    # Set global instance
    global _taste_graph_manager
    _taste_graph_manager = taste_graph_manager
    
    logger.info("Taste graph system initialized successfully")
    
    return taste_graph_manager
