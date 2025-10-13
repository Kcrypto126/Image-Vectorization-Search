# app/taste_graph.py
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json

from app.db import Board, PinTopic, Topic, User

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
        
    def initialize_user(self, user_id: str) -> Dict:
        """
        Initialize a new user in the taste graph system.
        Creates user record and initializes taste vector.
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
        
        return {
            'user_id': user_id,
            'taste_vector_version': 1,
            'status': 'initialized'
        }
    
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
        Rerank search results using user's taste vector and topic affinities.
        """
        if not user_id:
            return search_results[:top_k]
        
        # Get user taste vector
        user_taste = self.session.get_user_taste_vector(user_id)
        if user_taste is None:
            return search_results[:top_k]
        
        # Get user's top topics
        user_topics = dict(self.session.get_user_topics(user_id, 20))
        
        # Rerank results
        reranked_results = []
        
        for result in search_results:
            image_id = result['id']
            base_score = result['score']
            
            # Calculate taste similarity bonus
            taste_bonus = self._calculate_taste_similarity_bonus(user_taste, image_id)
            
            # Calculate topic affinity bonus
            topic_bonus = self._calculate_topic_affinity_bonus(user_topics, image_id)
            
            # Combine scores
            personalized_score = base_score + taste_bonus + topic_bonus
            
            reranked_results.append({
                **result,
                'score': personalized_score,
                'taste_bonus': taste_bonus,
                'topic_bonus': topic_bonus
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
        Create a new board for a user.
        """
        board = self.session.create_board(board_id, user_id, name, description)
        
        return {
            'board_id': board.board_id,
            'user_id': board.user_id,
            'name': board.name,
            'description': board.description,
            'status': 'created'
        }
    
    def add_pin_to_board(self, pin_id: str, board_id: str) -> Dict:
        """
        Add a pin to a board.
        """
        pin_board = self.session.add_pin_to_board(pin_id, board_id)
        
        return {
            'pin_id': pin_id,
            'board_id': board_id,
            'status': 'added'
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
