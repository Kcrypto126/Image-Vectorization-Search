# app/enhanced_auto_board_manager.py
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import json
import uuid

from app.db import ImageMetadata, Interaction, Board, PinBoard, UserCorrelation, PinCorrelation

logger = logging.getLogger(__name__)

class EnhancedAutoBoardManager:
    """
    Enhanced auto-board management system that implements:
    - Semantic similarity detection for new boards
    - Duplicate pin detection to prevent identical boards
    - Board expansion for same theme additions
    - Board decay for inactive boards
    - Board merging for highly similar boards
    """
    
    def __init__(self, session, vectorizer, index_adapter, 
                 min_pins_for_new_board: int = 15,
                 similarity_threshold: float = 0.7,
                 merge_threshold: float = 0.9,
                 decay_days: int = 30,
                 duplicate_threshold: float = 0.95):
        self.session = session
        self.vectorizer = vectorizer
        self.index_adapter = index_adapter
        self.min_pins_for_new_board = min_pins_for_new_board
        self.similarity_threshold = similarity_threshold
        self.merge_threshold = merge_threshold
        self.decay_days = decay_days
        self.duplicate_threshold = duplicate_threshold
    
    def process_user_interaction(self, user_id: str, image_id: str, event: str, weight: float = 1.0):
        """
        Process a user interaction and determine auto-board actions.
        This is the main entry point for the enhanced auto-board system.
        """
        logger.info(f"Processing {event} interaction for user {user_id} with image {image_id}")
        
        # Record the interaction
        self.session.record_interaction(user_id, image_id, event, weight)
        
        # Get user's recent interactions
        recent_interactions = self._get_recent_user_interactions(user_id, days=7)
        
        if len(recent_interactions) < self.min_pins_for_new_board:
            logger.info(f"User {user_id} has {len(recent_interactions)} recent interactions, not enough for auto-board analysis")
            return
        
        # Check for duplicate pins (identical content)
        if self._is_duplicate_content(user_id, image_id, recent_interactions):
            logger.info(f"Image {image_id} is duplicate content for user {user_id}, no auto-board action")
            return
        
        # Check if pins are semantically similar enough for a new board
        if self._are_pins_semantically_similar(recent_interactions):
            # Check if user already has a similar board
            similar_board = self._find_similar_existing_board(user_id, recent_interactions)
            
            if similar_board:
                # Expand existing board
                self._expand_board(similar_board, recent_interactions)
                logger.info(f"Expanded existing board {similar_board} for user {user_id}")
            else:
                # Create new auto-board
                board_id = self._create_semantic_auto_board(user_id, recent_interactions)
                if board_id:
                    logger.info(f"Created new semantic auto-board {board_id} for user {user_id}")
        else:
            logger.info(f"Recent interactions for user {user_id} are not semantically similar enough for auto-board")
        
        # Check for board merging opportunities
        self._check_and_merge_similar_boards(user_id)
        
        # Check for board decay
        self._check_and_decay_inactive_boards(user_id)
    
    def _get_recent_user_interactions(self, user_id: str, days: int = 7) -> List[Interaction]:
        """Get user's recent interactions"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        interactions = self.session.session.query(Interaction).filter(
            Interaction.user_id == user_id,
            Interaction.created_at >= cutoff_date
        ).order_by(Interaction.created_at.desc()).all()
        return interactions
    
    def _is_duplicate_content(self, user_id: str, image_id: str, recent_interactions: List[Interaction]) -> bool:
        """
        Check if the new image is too similar to recent interactions (duplicate content).
        Returns True if the image is essentially identical to recent pins.
        """
        try:
            new_vector = self.vectorizer.get_vector(image_id, self.index_adapter)
            if new_vector is None:
                return False
            
            for interaction in recent_interactions[:10]:  # Check last 10 interactions
                try:
                    existing_vector = self.vectorizer.get_vector(interaction.image_id, self.index_adapter)
                    if existing_vector is not None:
                        similarity = cosine_similarity([new_vector], [existing_vector])[0][0]
                        if similarity >= self.duplicate_threshold:
                            logger.info(f"Image {image_id} is {similarity:.3f} similar to {interaction.image_id} (duplicate)")
                            return True
                except Exception as e:
                    logger.warning(f"Failed to check similarity with {interaction.image_id}: {e}")
                    continue
            
            return False
        except Exception as e:
            logger.warning(f"Failed to check duplicate content for {image_id}: {e}")
            return False
    
    def _are_pins_semantically_similar(self, interactions: List[Interaction]) -> bool:
        """
        Check if recent interactions are semantically similar enough to warrant a new board.
        Returns True if the pins form a coherent semantic cluster.
        """
        if len(interactions) < self.min_pins_for_new_board:
            return False
        
        try:
            # Get vectors for recent interactions
            vectors = []
            for interaction in interactions:
                vector = self.vectorizer.get_vector(interaction.image_id, self.index_adapter)
                if vector is not None:
                    vectors.append(vector)
            
            if len(vectors) < self.min_pins_for_new_board:
                return False
            
            # Calculate pairwise similarities
            similarity_matrix = cosine_similarity(vectors)
            
            # Calculate average similarity (excluding diagonal)
            similarities = []
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    similarities.append(similarity_matrix[i][j])
            
            if not similarities:
                return False
            
            avg_similarity = np.mean(similarities)
            logger.info(f"Average semantic similarity of recent interactions: {avg_similarity:.3f}")
            
            return avg_similarity >= self.similarity_threshold
            
        except Exception as e:
            logger.warning(f"Failed to check semantic similarity: {e}")
            return False
    
    def _find_similar_existing_board(self, user_id: str, interactions: List[Interaction]) -> Optional[str]:
        """
        Find an existing auto-board that's similar to the recent interactions.
        Returns the board_id if found, None otherwise.
        """
        try:
            # Get user's existing auto-boards
            existing_boards = self.session.get_user_boards(user_id, board_type='auto', active_only=True)
            
            if not existing_boards:
                return None
            
            # Calculate average vector for recent interactions
            interaction_vectors = []
            for interaction in interactions:
                vector = self.vectorizer.get_vector(interaction.image_id, self.index_adapter)
                if vector is not None:
                    interaction_vectors.append(vector)
            
            if not interaction_vectors:
                return None
            
            avg_interaction_vector = np.mean(interaction_vectors, axis=0)
            
            # Compare with existing board vectors
            best_similarity = 0.0
            best_board = None
            
            for board in existing_boards:
                if board.board_vector:
                    try:
                        board_vector = np.array(board.board_vector)
                        similarity = cosine_similarity([avg_interaction_vector], [board_vector])[0][0]
                        
                        if similarity > best_similarity and similarity >= self.similarity_threshold:
                            best_similarity = similarity
                            best_board = board.board_id
                    except Exception as e:
                        logger.warning(f"Failed to compare with board {board.board_id}: {e}")
                        continue
            
            if best_board:
                logger.info(f"Found similar existing board {best_board} with similarity {best_similarity:.3f}")
            
            return best_board
            
        except Exception as e:
            logger.warning(f"Failed to find similar existing board: {e}")
            return None
    
    def _expand_board(self, board_id: str, interactions: List[Interaction]):
        """
        Expand an existing board with new similar interactions.
        """
        try:
            # Add new pins to the board
            for interaction in interactions:
                self.session.add_pin_to_board(interaction.image_id, board_id)
            
            # Update board vector
            self._update_board_vector(board_id)
            
            # Update board metadata
            board = self.session.session.query(Board).filter(Board.board_id == board_id).first()
            if board:
                board.last_activity = datetime.utcnow()
                board.pin_count = len(self.session.get_board_pins(board_id))
                self.session.session.commit()
            
            # Track board expansion
            self.session.track_board_evolution(
                board_id=board_id,
                evolution_type='expanded',
                reason=f"Added {len(interactions)} similar pins",
                evolution_data={'added_pins': [i.image_id for i in interactions]}
            )
            
            logger.info(f"Expanded board {board_id} with {len(interactions)} new pins")
            
        except Exception as e:
            logger.error(f"Failed to expand board {board_id}: {e}")
    
    def _create_semantic_auto_board(self, user_id: str, interactions: List[Interaction]) -> Optional[str]:
        """
        Create a new auto-board from semantically similar interactions.
        """
        try:
            # Get vectors for interactions
            vectors = []
            pin_ids = []
            for interaction in interactions:
                vector = self.vectorizer.get_vector(interaction.image_id, self.index_adapter)
                if vector is not None:
                    vectors.append(vector)
                    pin_ids.append(interaction.image_id)
            
            if not vectors:
                return None
            
            # Calculate board vector and cluster center
            board_vector = np.mean(vectors, axis=0)
            board_vector = board_vector / (np.linalg.norm(board_vector) + 1e-8)
            cluster_center = board_vector.copy()
            
            # Ensure all numpy types are converted to native Python types
            board_vector = board_vector.astype(np.float64).tolist()
            cluster_center = cluster_center.astype(np.float64).tolist()
            
            # Calculate cluster radius
            distances = [np.linalg.norm(vec - cluster_center) for vec in vectors]
            cluster_radius = float(np.mean(distances))
            
            # Generate board metadata
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
            
            # Update pin count
            board = self.session.session.query(Board).filter(Board.board_id == board_id).first()
            if board:
                board.pin_count = len(pin_ids)
                self.session.session.commit()
            
            # Track board creation
            self.session.track_board_evolution(
                board_id=board_id,
                evolution_type='created',
                reason=f"Auto-generated from {len(pin_ids)} semantically similar pins",
                evolution_data={'pin_count': len(pin_ids), 'avg_similarity': self._calculate_avg_similarity(vectors)}
            )
            
            return board_id
            
        except Exception as e:
            logger.error(f"Failed to create semantic auto-board: {e}")
            return None
    
    def _check_and_merge_similar_boards(self, user_id: str):
        """
        Check for highly similar boards and merge them if similarity > merge_threshold.
        """
        try:
            # Get user's active auto-boards
            boards = self.session.get_user_boards(user_id, board_type='auto', active_only=True)
            
            if len(boards) < 2:
                return
            
            # Calculate pairwise similarities
            board_vectors = []
            board_ids = []
            for board in boards:
                if board.board_vector:
                    board_vectors.append(np.array(board.board_vector))
                    board_ids.append(board.board_id)
            
            if len(board_vectors) < 2:
                return
            
            # Find pairs to merge
            to_merge = []
            for i in range(len(board_vectors)):
                for j in range(i + 1, len(board_vectors)):
                    similarity = cosine_similarity([board_vectors[i]], [board_vectors[j]])[0][0]
                    if similarity >= self.merge_threshold:
                        to_merge.append((board_ids[i], board_ids[j], similarity))
            
            # Merge similar boards
            for board_a, board_b, similarity in to_merge:
                self._merge_boards(board_a, board_b, similarity)
                
        except Exception as e:
            logger.error(f"Failed to check and merge similar boards for user {user_id}: {e}")
    
    def _merge_boards(self, board_a_id: str, board_b_id: str, similarity: float):
        """
        Merge two similar boards into one.
        """
        try:
            # Get pins from both boards
            pins_a = self.session.get_board_pins(board_a_id)
            pins_b = self.session.get_board_pins(board_b_id)
            
            # Choose the board with more pins as the target
            if len(pins_a) >= len(pins_b):
                target_board = board_a_id
                source_board = board_b_id
            else:
                target_board = board_b_id
                source_board = board_a_id
            
            # Move pins from source to target
            for pin_board in self.session.get_board_pins(source_board):
                # Check if pin is already in target board
                existing = self.session.session.query(PinBoard).filter(
                    PinBoard.pin_id == pin_board.pin_id,
                    PinBoard.board_id == target_board
                ).first()
                
                if not existing:
                    # Move pin to target board
                    pin_board.board_id = target_board
                    self.session.session.commit()
            
            # Update target board vector
            self._update_board_vector(target_board)
            
            # Deactivate source board
            source_board_obj = self.session.session.query(Board).filter(Board.board_id == source_board).first()
            if source_board_obj:
                source_board_obj.is_active = False
                self.session.session.commit()
            
            # Track board merge
            self.session.track_board_evolution(
                board_id=target_board,
                evolution_type='merged',
                previous_board_ids=[source_board],
                reason=f"Merged with board {source_board} (similarity: {similarity:.3f})",
                evolution_data={'similarity': similarity, 'merged_pins': len(pins_b)}
            )
            
            logger.info(f"Merged boards {source_board} into {target_board} (similarity: {similarity:.3f})")
            
        except Exception as e:
            logger.error(f"Failed to merge boards {board_a_id} and {board_b_id}: {e}")
    
    def _check_and_decay_inactive_boards(self, user_id: str):
        """
        Check for inactive boards and either decay their influence or archive them.
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.decay_days)
            
            # Get inactive auto-boards
            inactive_boards = self.session.session.query(Board).filter(
                Board.user_id == user_id,
                Board.board_type == 'auto',
                Board.is_active == True,
                Board.last_activity < cutoff_date
            ).all()
            
            for board in inactive_boards:
                # Check if board has very few pins (archive)
                pin_count = len(self.session.get_board_pins(board.board_id))
                if pin_count < 3:
                    self._archive_board(board.board_id, "Too few pins and inactive")
                else:
                    # Decay board influence by reducing its weight in recommendations
                    self._decay_board_influence(board.board_id)
                    
        except Exception as e:
            logger.error(f"Failed to check and decay inactive boards for user {user_id}: {e}")
    
    def _archive_board(self, board_id: str, reason: str):
        """
        Archive a board by marking it as inactive.
        """
        try:
            board = self.session.session.query(Board).filter(Board.board_id == board_id).first()
            if board:
                board.is_active = False
                self.session.session.commit()
                
                # Track board archival
                self.session.track_board_evolution(
                    board_id=board_id,
                    evolution_type='archived',
                    reason=reason,
                    evolution_data={'archived_at': datetime.utcnow().isoformat()}
                )
                
                logger.info(f"Archived board {board_id}: {reason}")
                
        except Exception as e:
            logger.error(f"Failed to archive board {board_id}: {e}")
    
    def _decay_board_influence(self, board_id: str):
        """
        Decay board influence by reducing its weight in recommendations.
        This could be implemented by storing a decay factor in the board metadata.
        """
        try:
            board = self.session.session.query(Board).filter(Board.board_id == board_id).first()
            if board:
                # For now, just update the last activity to reset the decay timer
                # In a more sophisticated implementation, you might store a decay factor
                board.last_activity = datetime.utcnow()
                self.session.session.commit()
                
                logger.info(f"Decayed influence for board {board_id}")
                
        except Exception as e:
            logger.error(f"Failed to decay board influence for {board_id}: {e}")
    
    def _update_board_vector(self, board_id: str):
        """
        Update a board's vector based on its current pins.
        """
        try:
            board_pins = self.session.get_board_pins(board_id)
            if not board_pins:
                return
            
            # Get vectors for all pins
            vectors = []
            for pin_board in board_pins:
                vector = self.vectorizer.get_vector(pin_board.pin_id, self.index_adapter)
                if vector is not None:
                    vectors.append(vector)
            
            if not vectors:
                return
            
            # Calculate new board vector
            board_vector = np.mean(vectors, axis=0)
            board_vector = board_vector / (np.linalg.norm(board_vector) + 1e-8)
            
            # Update board
            self.session.update_board_vector(board_id, board_vector)
            
        except Exception as e:
            logger.error(f"Failed to update board vector for {board_id}: {e}")
    
    def _generate_board_metadata(self, pin_ids: List[str], cluster_center: np.ndarray) -> Tuple[str, str]:
        """
        Generate name and description for a board based on its pins.
        """
        try:
            # For now, generate generic metadata
            # In a more sophisticated implementation, you might use topic modeling
            # or image analysis to generate more descriptive names
            
            board_name = f"Auto Board {len(pin_ids)} Pins"
            board_description = f"Auto-generated board with {len(pin_ids)} semantically similar pins"
            
            return board_name, board_description
            
        except Exception as e:
            logger.warning(f"Failed to generate board metadata: {e}")
            return "Auto Board", "Auto-generated board"
    
    def _calculate_avg_similarity(self, vectors: List[np.ndarray]) -> float:
        """
        Calculate average pairwise similarity for a set of vectors.
        """
        try:
            if len(vectors) < 2:
                return 1.0
            
            similarities = []
            for i in range(len(vectors)):
                for j in range(i + 1, len(vectors)):
                    similarity = cosine_similarity([vectors[i]], [vectors[j]])[0][0]
                    similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.warning(f"Failed to calculate average similarity: {e}")
            return 0.0
