# app/correlation_engine.py
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
import itertools

from app.db import Interaction, User, ImageMetadata, UserCorrelation, PinCorrelation, ItemCooccurrence

logger = logging.getLogger(__name__)

class CorrelationEngine:
    """
    Engine for computing and managing user-user and pin-pin correlations.
    Establishes behavioral patterns and similarity relationships.
    """
    
    def __init__(self, session, vectorizer, index_adapter, 
                 min_interactions: int = 5, similarity_threshold: float = 0.3):
        self.session = session
        self.vectorizer = vectorizer
        self.index_adapter = index_adapter
        self.min_interactions = min_interactions
        self.similarity_threshold = similarity_threshold
    
    def compute_user_correlations(self, user_id: str, max_users: int = 100) -> List[Tuple[str, float, float]]:
        """
        Compute correlations between a user and other users.
        Returns list of (user_id, similarity_score, interaction_overlap)
        """
        logger.info(f"Computing user correlations for {user_id}")
        
        # Get user's taste vector
        user_taste = self.session.get_user_taste_vector(user_id)
        if user_taste is None:
            logger.warning(f"No taste vector found for user {user_id}")
            return []
        
        # Get user's interaction history
        user_interactions = self.session.get_user_interaction_history(user_id, limit=1000)
        user_pin_ids = set(interaction.image_id for interaction in user_interactions)
        
        if len(user_pin_ids) < self.min_interactions:
            logger.info(f"Not enough interactions for user {user_id}")
            return []
        
        # Get other users with sufficient interactions
        other_users = self._get_users_with_interactions(min_interactions=self.min_interactions)
        other_users = [u for u in other_users if u != user_id]
        
        correlations = []
        
        for other_user_id in other_users[:max_users]:
            try:
                # Get other user's taste vector
                other_taste = self.session.get_user_taste_vector(other_user_id)
                if other_taste is None:
                    continue
                
                # Calculate taste similarity
                similarity = np.dot(user_taste, other_taste)
                
                # Calculate interaction overlap
                other_interactions = self.session.get_user_interaction_history(other_user_id, limit=1000)
                other_pin_ids = set(interaction.image_id for interaction in other_interactions)
                
                overlap = len(user_pin_ids & other_pin_ids)
                total_interactions = len(user_pin_ids | other_pin_ids)
                interaction_overlap = overlap / total_interactions if total_interactions > 0 else 0
                
                # Only keep significant correlations
                if similarity >= self.similarity_threshold or interaction_overlap >= 0.1:
                    correlations.append((other_user_id, similarity, interaction_overlap))
                    
                    # Store in database
                    self.session.update_user_correlation(
                        user_id, other_user_id, similarity, interaction_overlap
                    )
                    
            except Exception as e:
                logger.warning(f"Failed to compute correlation for user {other_user_id}: {e}")
                continue
        
        # Sort by combined score
        correlations.sort(key=lambda x: x[1] * 0.7 + x[2] * 0.3, reverse=True)
        
        logger.info(f"Computed {len(correlations)} user correlations for {user_id}")
        return correlations
    
    def compute_pin_correlations(self, pin_id: str, max_pins: int = 200) -> List[Tuple[str, float, float, float]]:
        """
        Compute correlations between a pin and other pins.
        Returns list of (pin_id, similarity_score, cooccurrence_score, behavioral_score)
        """
        logger.info(f"Computing pin correlations for {pin_id}")
        
        # Get pin vector
        pin_vector = self.vectorizer.get_vector(pin_id, self.index_adapter)
        if pin_vector is None:
            logger.warning(f"No vector found for pin {pin_id}")
            return []
        
        # Get co-occurrence data
        cooccurrence_data = self._get_pin_cooccurrences(pin_id)
        
        # Get behavioral data (users who interacted with this pin)
        behavioral_data = self._get_pin_behavioral_data(pin_id)
        
        # Get other pins to compare with
        other_pins = self._get_pins_for_correlation(pin_id, max_pins)
        
        correlations = []
        
        for other_pin_id in other_pins:
            try:
                # Get other pin vector
                other_vector = self.vectorizer.get_vector(other_pin_id, self.index_adapter)
                if other_vector is None:
                    continue
                
                # Calculate vector similarity
                similarity = np.dot(pin_vector, other_vector)
                
                # Calculate co-occurrence score
                cooccurrence_score = cooccurrence_data.get(other_pin_id, 0.0)
                
                # Calculate behavioral score
                behavioral_score = behavioral_data.get(other_pin_id, 0.0)
                
                # Only keep significant correlations
                if (similarity >= self.similarity_threshold or 
                    cooccurrence_score >= 0.1 or 
                    behavioral_score >= 0.1):
                    
                    correlations.append((other_pin_id, similarity, cooccurrence_score, behavioral_score))
                    
                    # Store in database
                    self.session.update_pin_correlation(
                        pin_id, other_pin_id, similarity, cooccurrence_score, behavioral_score
                    )
                    
            except Exception as e:
                logger.warning(f"Failed to compute correlation for pin {other_pin_id}: {e}")
                continue
        
        # Sort by combined score
        correlations.sort(key=lambda x: x[1] * 0.5 + x[2] * 0.3 + x[3] * 0.2, reverse=True)
        
        logger.info(f"Computed {len(correlations)} pin correlations for {pin_id}")
        return correlations
    
    def _get_users_with_interactions(self, min_interactions: int = 5) -> List[str]:
        """Get users with sufficient interaction history"""
        # Get users with at least min_interactions
        result = self.session.session.query(Interaction.user_id).group_by(
            Interaction.user_id
        ).having(
            self.session.session.func.count(Interaction.id) >= min_interactions
        ).all()
        
        return [row[0] for row in result]
    
    def _get_pin_cooccurrences(self, pin_id: str) -> Dict[str, float]:
        """Get co-occurrence scores for a pin"""
        cooccurrences = self.session.session.query(ItemCooccurrence).filter(
            (ItemCooccurrence.item_a == pin_id) | (ItemCooccurrence.item_b == pin_id)
        ).all()
        
        cooccurrence_data = {}
        for cooc in cooccurrences:
            other_pin = cooc.item_b if cooc.item_a == pin_id else cooc.item_a
            cooccurrence_data[other_pin] = cooc.score
        
        return cooccurrence_data
    
    def _get_pin_behavioral_data(self, pin_id: str) -> Dict[str, float]:
        """Get behavioral correlation data for a pin"""
        # Get users who interacted with this pin
        interactions = self.session.session.query(Interaction).filter(
            Interaction.image_id == pin_id
        ).all()
        
        user_ids = [interaction.user_id for interaction in interactions]
        
        # For each user, find other pins they interacted with
        behavioral_scores = defaultdict(float)
        
        for user_id in user_ids:
            user_interactions = self.session.session.query(Interaction).filter(
                Interaction.user_id == user_id,
                Interaction.image_id != pin_id
            ).all()
            
            for interaction in user_interactions:
                other_pin_id = interaction.image_id
                # Weight by interaction strength
                weight = self._get_interaction_weight(interaction.event, interaction.weight)
                behavioral_scores[other_pin_id] += weight
        
        # Normalize scores
        if behavioral_scores:
            max_score = max(behavioral_scores.values())
            behavioral_scores = {k: v / max_score for k, v in behavioral_scores.items()}
        
        return dict(behavioral_scores)
    
    def _get_pins_for_correlation(self, pin_id: str, max_pins: int) -> List[str]:
        """Get pins to compare with for correlation"""
        # Get pins from co-occurrence data
        cooccurrence_pins = self._get_pin_cooccurrences(pin_id)
        
        # Get pins from behavioral data
        behavioral_pins = self._get_pin_behavioral_data(pin_id)
        
        # Combine and deduplicate
        all_pins = set(cooccurrence_pins.keys()) | set(behavioral_pins.keys())
        all_pins.discard(pin_id)  # Remove self
        
        return list(all_pins)[:max_pins]
    
    def _get_interaction_weight(self, event: str, base_weight: float) -> float:
        """Convert interaction event to weight"""
        event_weights = {
            'save': 2.0,
            'like': 1.5,
            'click': 1.0,
            'view': 0.5
        }
        return event_weights.get(event, 1.0) * base_weight
    
    def get_recommendations_for_user(self, user_id: str, top_k: int = 20) -> List[Tuple[str, float, str]]:
        """
        Get recommendations for a user based on similar users' preferences.
        Returns list of (pin_id, score, reason)
        """
        logger.info(f"Getting recommendations for user {user_id}")
        
        # Get similar users
        similar_users = self.session.get_similar_users(user_id, top_k=10)
        if not similar_users:
            return []
        
        # Get pins liked by similar users
        recommendation_scores = defaultdict(float)
        recommendation_reasons = {}
        
        for similar_user_id, similarity, overlap in similar_users:
            # Get pins this similar user liked
            similar_user_interactions = self.session.get_user_interaction_history(similar_user_id, limit=100)
            
            for interaction in similar_user_interactions:
                pin_id = interaction.image_id
                weight = self._get_interaction_weight(interaction.event, interaction.weight)
                
                # Score based on user similarity and interaction strength
                score = similarity * weight * (1 + overlap)
                recommendation_scores[pin_id] += score
                
                # Track reason
                if pin_id not in recommendation_reasons:
                    recommendation_reasons[pin_id] = f"Liked by similar user {similar_user_id}"
        
        # Get user's existing interactions to filter out
        user_interactions = self.session.get_user_interaction_history(user_id, limit=1000)
        user_pin_ids = set(interaction.image_id for interaction in user_interactions)
        
        # Filter out pins user already interacted with
        recommendations = []
        for pin_id, score in recommendation_scores.items():
            if pin_id not in user_pin_ids:
                recommendations.append((pin_id, score, recommendation_reasons.get(pin_id, "Similar user preference")))
        
        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
        return recommendations[:top_k]
    
    def get_similar_content_for_pin(self, pin_id: str, top_k: int = 20) -> List[Tuple[str, float, str]]:
        """
        Get similar content for a pin based on correlations.
        Returns list of (pin_id, score, reason)
        """
        logger.info(f"Getting similar content for pin {pin_id}")
        
        # Get similar pins from correlations
        similar_pins = self.session.get_similar_pins(pin_id, top_k=50)
        
        recommendations = []
        for other_pin_id, similarity, cooccurrence, behavioral in similar_pins:
            # Combine different correlation types
            combined_score = similarity * 0.5 + cooccurrence * 0.3 + behavioral * 0.2
            
            # Determine reason
            if similarity > cooccurrence and similarity > behavioral:
                reason = "Visual similarity"
            elif cooccurrence > behavioral:
                reason = "Often appear together"
            else:
                reason = "Similar user behavior"
            
            recommendations.append((other_pin_id, combined_score, reason))
        
        # Sort by combined score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Generated {len(recommendations)} similar content for pin {pin_id}")
        return recommendations[:top_k]
    
    def update_correlations_batch(self, user_ids: List[str] = None, pin_ids: List[str] = None):
        """Update correlations for multiple users/pins in batch"""
        if user_ids:
            logger.info(f"Updating user correlations for {len(user_ids)} users")
            for user_id in user_ids:
                try:
                    self.compute_user_correlations(user_id, max_users=50)
                except Exception as e:
                    logger.warning(f"Failed to update correlations for user {user_id}: {e}")
        
        if pin_ids:
            logger.info(f"Updating pin correlations for {len(pin_ids)} pins")
            for pin_id in pin_ids:
                try:
                    self.compute_pin_correlations(pin_id, max_pins=100)
                except Exception as e:
                    logger.warning(f"Failed to update correlations for pin {pin_id}: {e}")
    
    def get_correlation_stats(self) -> Dict[str, int]:
        """Get statistics about computed correlations"""
        user_correlations_count = self.session.session.query(UserCorrelation).count()
        pin_correlations_count = self.session.session.query(PinCorrelation).count()
        
        return {
            'user_correlations': user_correlations_count,
            'pin_correlations': pin_correlations_count
        }
