# app/auto_board_creator.py
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import json

from app.db import ImageMetadata, Interaction, Board, PinBoard, UserCorrelation, PinCorrelation

logger = logging.getLogger(__name__)

class AutoBoardCreator:
    """
    Creates and manages auto-generated boards based on user behavior patterns.
    Uses clustering algorithms to identify taste clusters and create boards.
    """
    
    def __init__(self, session, vectorizer, index_adapter, min_pins_per_board: int = 5,
                 max_boards_per_user: int = 20, similarity_threshold: float = 0.7):
        self.session = session
        self.vectorizer = vectorizer
        self.index_adapter = index_adapter
        self.min_pins_per_board = min_pins_per_board
        self.max_boards_per_user = max_boards_per_user
        self.similarity_threshold = similarity_threshold
        
    def is_user_ready_for_auto_boards(self, user_id: str) -> bool:
        """
        Check if a user has enough interactions to create auto-boards.
        """
        recent_interactions = self._get_recent_user_interactions(user_id, days=30)
        return len(recent_interactions) >= self.min_pins_per_board
    
    def create_auto_boards_for_user(self, user_id: str, force_recreate: bool = False) -> List[str]:
        """
        Create auto-boards for a user based on their interaction patterns.
        """
        logger.info(f"Creating auto-boards for user {user_id}")
        
        # Check if user is ready for auto-boards
        if not self.is_user_ready_for_auto_boards(user_id):
            logger.info(f"User {user_id} not ready for auto-boards (insufficient interactions)")
            return []
        
        # Get user's recent interactions
        recent_interactions = self._get_recent_user_interactions(user_id, days=30)
        
        # Get vectors for interacted pins
        pin_vectors = self._get_pin_vectors(recent_interactions)
        if not pin_vectors:
            logger.warning(f"No vectors found for user {user_id}'s interactions")
            return []
        
        # Cluster the pins
        clusters = self._cluster_pins(pin_vectors)
        if not clusters:
            logger.info(f"No meaningful clusters found for user {user_id}")
            return []
        
        # Create boards for each cluster
        created_boards = []
        for cluster_id, cluster_pins in clusters.items():
            if len(cluster_pins) >= self.min_pins_per_board:
                board_id = self._create_board_from_cluster(user_id, cluster_id, cluster_pins)
                if board_id:
                    created_boards.append(board_id)
        
        # Limit number of auto-boards
        if len(created_boards) > self.max_boards_per_user:
            created_boards = created_boards[:self.max_boards_per_user]
        
        logger.info(f"Created {len(created_boards)} auto-boards for user {user_id}")
        return created_boards
    
    def _get_recent_user_interactions(self, user_id: str, days: int = 30) -> List[Interaction]:
        """Get user's recent interactions"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        interactions = self.session.session.query(Interaction).filter(
            Interaction.user_id == user_id,
            Interaction.created_at >= cutoff_date
        ).order_by(Interaction.created_at.desc()).all()
        return interactions
    
    def _get_pin_vectors(self, interactions: List[Interaction]) -> Dict[str, np.ndarray]:
        """Get vectors for pins from interactions"""
        pin_vectors = {}
        pin_ids = [interaction.image_id for interaction in interactions]
        
        for pin_id in pin_ids:
            try:
                vector = self.vectorizer.get_vector(pin_id, self.index_adapter)
                if vector is not None:
                    pin_vectors[pin_id] = vector
            except Exception as e:
                logger.warning(f"Failed to get vector for pin {pin_id}: {e}")
        
        return pin_vectors
    
    def _cluster_pins(self, pin_vectors: Dict[str, np.ndarray]) -> Dict[int, List[str]]:
        """Cluster pins using K-means and DBSCAN"""
        if len(pin_vectors) < self.min_pins_per_board:
            return {}
        
        # Convert to matrix
        pin_ids = list(pin_vectors.keys())
        vectors = np.array([pin_vectors[pin_id] for pin_id in pin_ids])
        
        # Try different clustering approaches
        clusters = {}
        
        # Method 1: K-means with optimal k
        optimal_k = min(max(2, len(pin_ids) // self.min_pins_per_board), 10)
        try:
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(vectors)
            
            # Group pins by cluster
            for i, label in enumerate(kmeans_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(pin_ids[i])
            
            # Filter out small clusters
            clusters = {k: v for k, v in clusters.items() if len(v) >= self.min_pins_per_board}
            
        except Exception as e:
            logger.warning(f"K-means clustering failed: {e}")
        
        # Method 2: DBSCAN for density-based clustering
        if not clusters:
            try:
                dbscan = DBSCAN(eps=0.3, min_samples=self.min_pins_per_board)
                dbscan_labels = dbscan.fit_predict(vectors)
                
                # Group pins by cluster
                for i, label in enumerate(dbscan_labels):
                    if label != -1:  # -1 is noise in DBSCAN
                        if label not in clusters:
                            clusters[label] = []
                        clusters[label].append(pin_ids[i])
                        
            except Exception as e:
                logger.warning(f"DBSCAN clustering failed: {e}")
        
        return clusters
    
    def _create_board_from_cluster(self, user_id: str, cluster_id: int, pin_ids: List[str]) -> Optional[str]:
        """Create a board from a cluster of pins"""
        try:
            # Get vectors for cluster pins
            cluster_vectors = []
            for pin_id in pin_ids:
                vector = self.vectorizer.get_vector(pin_id, self.index_adapter)
                if vector is not None:
                    cluster_vectors.append(vector)
            
            if not cluster_vectors:
                return None
            
            # Calculate cluster center and board vector
            cluster_center = np.mean(cluster_vectors, axis=0)
            board_vector = cluster_center / (np.linalg.norm(cluster_center) + 1e-8)
            
            # Calculate cluster radius
            distances = [np.linalg.norm(vec - cluster_center) for vec in cluster_vectors]
            cluster_radius = float(np.mean(distances))
            
            # Generate board name and description
            board_name, board_description = self._generate_board_metadata(pin_ids, cluster_center)
            
            # Create the board
            board_id = self.session.create_auto_board(
                user_id=user_id,
                name=board_name,
                description=board_description,
                board_vector=board_vector,
                cluster_center=cluster_center,
                cluster_radius=cluster_radius
            )
            
            # Add pins to the board
            for pin_id in pin_ids:
                self.session.add_pin_to_board(pin_id, board_id)
            
            # Track board creation
            self.session.track_board_evolution(
                board_id=board_id,
                evolution_type='created',
                reason=f"Auto-generated from {len(pin_ids)} similar pins",
                evolution_data={'cluster_id': cluster_id, 'pin_count': len(pin_ids)}
            )
            
            logger.info(f"Created auto-board {board_id} with {len(pin_ids)} pins")
            return board_id
            
        except Exception as e:
            logger.error(f"Failed to create board from cluster {cluster_id}: {e}")
            return None
    
    def _generate_board_metadata(self, pin_ids: List[str], cluster_center: np.ndarray) -> Tuple[str, str]:
        """Generate name and description for auto-board based on cluster characteristics"""
        # Get metadata for pins to understand the theme
        pin_metadata = []
        for pin_id in pin_ids[:10]:  # Sample first 10 pins
            try:
                metadata = self.session.session.query(ImageMetadata).filter(
                    ImageMetadata.id == pin_id
                ).first()
                if metadata and metadata.generated_metadata:
                    pin_metadata.append(metadata.generated_metadata)
            except Exception:
                continue
        
        # Extract common themes
        themes = self._extract_common_themes(pin_metadata)
        
        # Generate name and description
        if themes:
            board_name = f"Auto: {themes[0]}"
            board_description = f"Auto-generated board featuring {', '.join(themes[:3])}"
        else:
            board_name = f"Auto Board {datetime.utcnow().strftime('%Y%m%d')}"
            board_description = "Auto-generated board based on your recent interests"
        
        return board_name, board_description
    
    def _extract_common_themes(self, pin_metadata: List[Dict]) -> List[str]:
        """Extract common themes from pin metadata"""
        all_categories = []
        all_keywords = []
        all_styles = []
        
        for metadata in pin_metadata:
            if isinstance(metadata, dict):
                categories = metadata.get('Categories', [])
                keywords = metadata.get('Keywords', [])
                styles = metadata.get('Visual style', [])
                
                if isinstance(categories, list):
                    all_categories.extend(categories)
                if isinstance(keywords, list):
                    all_keywords.extend(keywords)
                if isinstance(styles, list):
                    all_styles.extend(styles)
        
        # Find most common themes
        category_counts = Counter(all_categories)
        keyword_counts = Counter(all_keywords)
        style_counts = Counter(all_styles)
        
        themes = []
        themes.extend([cat for cat, count in category_counts.most_common(3) if count > 1])
        themes.extend([kw for kw, count in keyword_counts.most_common(3) if count > 1])
        themes.extend([style for style, count in style_counts.most_common(3) if count > 1])
        
        return list(set(themes))[:5]  # Return top 5 unique themes
    
    def update_auto_boards(self, user_id: str) -> List[str]:
        """Update existing auto-boards based on new interactions"""
        logger.info(f"Updating auto-boards for user {user_id}")
        
        # Get existing auto-boards
        auto_boards = self.session.get_user_boards(user_id, board_type='auto')
        
        # Get recent interactions
        recent_interactions = self._get_recent_user_interactions(user_id, days=7)
        if not recent_interactions:
            return []
        
        updated_boards = []
        
        for board in auto_boards:
            # Check if new pins fit into existing boards
            new_pins = self._find_pins_for_existing_board(board, recent_interactions)
            if new_pins:
                # Add new pins to board
                for pin_id in new_pins:
                    self.session.add_pin_to_board(pin_id, board.board_id)
                
                # Update board vector
                self._update_board_vector(board.board_id)
                updated_boards.append(board.board_id)
        
        # Create new boards if needed
        remaining_interactions = self._get_remaining_interactions(auto_boards, recent_interactions)
        if remaining_interactions:
            new_boards = self.create_auto_boards_for_user(user_id)
            updated_boards.extend(new_boards)
        
        return updated_boards
    
    def _find_pins_for_existing_board(self, board: Board, interactions: List[Interaction]) -> List[str]:
        """Find pins from recent interactions that fit into an existing board"""
        if not board.board_vector or not board.cluster_center:
            return []
        
        board_vector = np.array(board.board_vector)
        cluster_center = np.array(board.cluster_center)
        cluster_radius = board.cluster_radius or 0.5
        
        fitting_pins = []
        
        for interaction in interactions:
            pin_id = interaction.image_id
            try:
                pin_vector = self.vectorizer.get_vector(pin_id, self.index_adapter)
                if pin_vector is not None:
                    # Check if pin is within cluster radius
                    distance = np.linalg.norm(pin_vector - cluster_center)
                    if distance <= cluster_radius * 1.5:  # Allow some expansion
                        fitting_pins.append(pin_id)
            except Exception:
                continue
        
        return fitting_pins
    
    def _get_remaining_interactions(self, existing_boards: List[Board], interactions: List[Interaction]) -> List[Interaction]:
        """Get interactions that don't fit into existing boards"""
        board_pin_ids = set()
        for board in existing_boards:
            board_pins = self.session.get_board_pins(board.board_id)
            board_pin_ids.update([pb.pin_id for pb in board_pins])
        
        remaining = [interaction for interaction in interactions 
                    if interaction.image_id not in board_pin_ids]
        return remaining
    
    def _update_board_vector(self, board_id: str):
        """Update board vector based on current pins"""
        board_pins = self.session.get_board_pins(board_id)
        if not board_pins:
            return
        
        # Get vectors for all pins in board
        pin_vectors = []
        for pin_board in board_pins:
            try:
                vector = self.vectorizer.get_vector(pin_board.pin_id, self.index_adapter)
                if vector is not None:
                    pin_vectors.append(vector)
            except Exception:
                continue
        
        if pin_vectors:
            # Update board vector
            new_board_vector = np.mean(pin_vectors, axis=0)
            new_board_vector = new_board_vector / (np.linalg.norm(new_board_vector) + 1e-8)
            self.session.update_board_vector(board_id, new_board_vector)
            
            # Update cluster center and radius
            board = self.session.session.query(Board).filter(Board.board_id == board_id).first()
            if board:
                new_cluster_center = np.mean(pin_vectors, axis=0)
                distances = [np.linalg.norm(vec - new_cluster_center) for vec in pin_vectors]
                new_cluster_radius = float(np.mean(distances))
                
                board.cluster_center = new_cluster_center.tolist()
                board.cluster_radius = new_cluster_radius
                board.pin_count = len(pin_vectors)
                self.session.session.commit()

class BoardVectorManager:
    """
    Manages board vector generation and updates.
    """
    
    def __init__(self, session, vectorizer, index_adapter):
        self.session = session
        self.vectorizer = vectorizer
        self.index_adapter = index_adapter
    
    def generate_board_vector(self, board_id: str) -> Optional[np.ndarray]:
        """Generate vector for a board based on its pins"""
        board_pins = self.session.get_board_pins(board_id)
        if not board_pins:
            return None
        
        # Get vectors for all pins
        pin_vectors = []
        for pin_board in board_pins:
            try:
                vector = self.vectorizer.get_vector(pin_board.pin_id, self.index_adapter)
                if vector is not None:
                    pin_vectors.append(vector)
            except Exception:
                continue
        
        if not pin_vectors:
            return None
        
        # Calculate board vector as weighted average
        board_vector = np.mean(pin_vectors, axis=0)
        board_vector = board_vector / (np.linalg.norm(board_vector) + 1e-8)
        
        return board_vector
    
    def update_all_board_vectors(self):
        """Update vectors for all boards"""
        boards = self.session.session.query(Board).filter(Board.is_active == True).all()
        
        for board in boards:
            try:
                board_vector = self.generate_board_vector(board.board_id)
                if board_vector is not None:
                    self.session.update_board_vector(board.board_id, board_vector)
                    logger.info(f"Updated vector for board {board.board_id}")
            except Exception as e:
                logger.warning(f"Failed to update vector for board {board.board_id}: {e}")
