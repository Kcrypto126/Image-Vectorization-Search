# app/pinsage.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
import logging
from datetime import datetime, timedelta
import math

from app.db import ImageMetadata, Interaction, ItemCooccurrence, PinTopic, Topic, User, UserTopic

logger = logging.getLogger(__name__)

class PinSageLayer(nn.Module):
    """
    PinSage convolution layer implementation.
    Aggregates neighbor information using weighted sampling and attention mechanisms.
    """
    def __init__(self, input_dim: int, output_dim: int, num_samples: int = 25):
        super(PinSageLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_samples = num_samples
        
        # Convolution weights
        self.conv_weight = nn.Linear(input_dim, output_dim, bias=False)
        self.self_weight = nn.Linear(input_dim, output_dim, bias=False)
        
        # Attention mechanism
        self.attention_weight = nn.Linear(output_dim, 1, bias=False)
        
        # Normalization
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, node_features: torch.Tensor, neighbor_features: torch.Tensor, 
                neighbor_weights: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PinSage layer.
        
        Args:
            node_features: Current node features [batch_size, input_dim]
            neighbor_features: Sampled neighbor features [batch_size, num_samples, input_dim]
            neighbor_weights: Edge weights for neighbors [batch_size, num_samples]
        """
        batch_size = node_features.size(0)
        
        # Transform node features
        self_features = self.self_weight(node_features)  # [batch_size, output_dim]
        
        # Transform neighbor features
        neighbor_transformed = self.conv_weight(neighbor_features)  # [batch_size, num_samples, output_dim]
        
        # Compute attention scores
        attention_scores = self.attention_weight(neighbor_transformed).squeeze(-1)  # [batch_size, num_samples]
        
        # Apply edge weights
        attention_scores = attention_scores * neighbor_weights
        
        # Normalize attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, num_samples]
        
        # Aggregate neighbor information
        aggregated = torch.sum(neighbor_transformed * attention_weights.unsqueeze(-1), dim=1)  # [batch_size, output_dim]
        
        # Combine with self features
        output = self_features + aggregated
        
        # Apply normalization and activation
        output = self.norm(output)
        output = F.relu(output)
        
        return output

class PinSageModel(nn.Module):
    """
    Complete PinSage model for learning user taste representations.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 num_samples: int = 25, dropout: float = 0.1):
        super(PinSageModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_samples = num_samples
        
        # Build layers
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList([
            PinSageLayer(dims[i], dims[i+1], num_samples) 
            for i in range(len(dims)-1)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.final_projection = nn.Linear(output_dim, output_dim)
        
    def forward(self, node_features: torch.Tensor, neighbor_features: torch.Tensor, 
                neighbor_weights: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all PinSage layers.
        """
        x = node_features
        
        for layer in self.layers:
            x = layer(x, neighbor_features, neighbor_weights)
            x = self.dropout(x)
        
        # Final projection
        x = self.final_projection(x)
        
        return x

class TasteGraphBuilder:
    """
    Builds and manages the taste graph structure for PinSage training.
    """
    def __init__(self, session, vectorizer, index_adapter=None):
        self.session = session
        self.vectorizer = vectorizer
        self.index_adapter = index_adapter
        self.graph = nx.MultiDiGraph()
        self.node_features = {}
        self.edge_weights = {}
        
    def build_graph(self, user_ids: List[str] = None, max_interactions: int = 10000):
        """
        Build the taste graph from user interactions and content relationships.
        """
        logger.info("Building taste graph...")
        
        # Get all users or specific users
        if user_ids:
            users = [self.session.get_or_create_user(uid) for uid in user_ids]
        else:
            users = self.session.session.query(User).all()
        
        logger.info(f"Processing {len(users)} users")
        
        # Add user nodes
        for user in users:
            self._add_user_node(user)
        
        # Add content nodes (pins/images)
        self._add_content_nodes()
        
        # Add topic nodes
        self._add_topic_nodes()
        
        # Add interaction edges
        self._add_interaction_edges(max_interactions)
        
        # Add similarity edges
        self._add_similarity_edges()
        
        # Add topic affinity edges
        self._add_topic_affinity_edges()
        
        logger.info(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
    def _add_user_node(self, user):
        """Add user node to graph"""
        node_id = f"user_{user.user_id}"
        self.graph.add_node(node_id, type='user', user_id=user.user_id)
        
        # Initialize user features if not present
        if user.taste_vector:
            self.node_features[node_id] = np.array(user.taste_vector)
        else:
            # Initialize with random vector
            self.node_features[node_id] = np.random.normal(0, 0.1, 512)
    
    def _add_content_nodes(self):
        """Add content (pin/image) nodes to graph"""
        images = self.session.session.query(ImageMetadata).all()
        
        for img in images:
            node_id = f"pin_{img.id}"
            self.graph.add_node(node_id, type='pin', image_id=img.id)
            
            # Get exact image vector from index if available
            if self.index_adapter:
                try:
                    image_vector = self.vectorizer.get_vector(img.id, self.index_adapter)
                    if image_vector is not None:
                        self.node_features[node_id] = image_vector
                        continue
                except Exception as e:
                    logger.warning(f"Failed to get vector for image {img.id}: {e}")
            
            # Fallback to random initialization if index not available or vector not found
            self.node_features[node_id] = np.random.normal(0, 0.1, 512)
    
    def _add_topic_nodes(self):
        """Add topic nodes to graph"""
        topics = self.session.session.query(Topic).all()
        
        for topic in topics:
            node_id = f"topic_{topic.topic_id}"
            self.graph.add_node(node_id, type='topic', topic_id=topic.topic_id)
            
            if topic.topic_vector:
                self.node_features[node_id] = np.array(topic.topic_vector)
            else:
                # Initialize topic vector based on associated pins
                self.node_features[node_id] = np.random.normal(0, 0.1, 512)
    
    def _add_interaction_edges(self, max_interactions: int):
        """Add user-pin interaction edges"""
        interactions = self.session.session.query(Interaction).limit(max_interactions).all()
        
        for interaction in interactions:
            user_node = f"user_{interaction.user_id}"
            pin_node = f"pin_{interaction.image_id}"
            
            if user_node in self.graph and pin_node in self.graph:
                # Weight based on interaction type and strength
                weight = self._get_interaction_weight(interaction.event, interaction.weight)
                
                self.graph.add_edge(
                    user_node, pin_node, 
                    type='interaction', 
                    weight=weight,
                    event=interaction.event,
                    timestamp=interaction.created_at
                )
                
                # Add reverse edge for bidirectional relationships
                self.graph.add_edge(
                    pin_node, user_node,
                    type='interaction_reverse',
                    weight=weight * 0.5,  # Lower weight for reverse
                    event=interaction.event,
                    timestamp=interaction.created_at
                )
    
    def _add_similarity_edges(self):
        """Add similarity edges between similar content"""
        # Use existing co-occurrence data
        cooccurrences = self.session.session.query(ItemCooccurrence).all()
        
        for cooc in cooccurrences:
            pin1_node = f"pin_{cooc.item_a}"
            pin2_node = f"pin_{cooc.item_b}"
            
            if pin1_node in self.graph and pin2_node in self.graph:
                self.graph.add_edge(
                    pin1_node, pin2_node,
                    type='similarity',
                    weight=cooc.score
                )
                self.graph.add_edge(
                    pin2_node, pin1_node,
                    type='similarity',
                    weight=cooc.score
                )
    
    def _add_topic_affinity_edges(self):
        """Add user-topic affinity edges"""
        user_topics = self.session.session.query(UserTopic).all()
        
        for ut in user_topics:
            user_node = f"user_{ut.user_id}"
            topic_node = f"topic_{ut.topic_id}"
            
            if user_node in self.graph and topic_node in self.graph:
                self.graph.add_edge(
                    user_node, topic_node,
                    type='topic_affinity',
                    weight=ut.affinity_score,
                    interaction_count=ut.interaction_count
                )
    
    def _get_interaction_weight(self, event: str, base_weight: float) -> float:
        """Convert interaction event to edge weight"""
        event_weights = {
            'save': 2.0,
            'like': 1.5,
            'click': 1.0,
            'view': 0.5
        }
        return event_weights.get(event, 1.0) * base_weight
    
    def sample_neighbors(self, node_id: str, num_samples: int = 25) -> Tuple[List[str], List[float]]:
        """
        Sample neighbors for a node using weighted random sampling.
        """
        if node_id not in self.graph:
            return [], []
        
        neighbors = list(self.graph.neighbors(node_id))
        if not neighbors:
            return [], []
        
        # Get edge weights
        weights = []
        for neighbor in neighbors:
            edge_data = self.graph.get_edge_data(node_id, neighbor)
            if edge_data:
                # Sum weights across multiple edges
                total_weight = sum(data.get('weight', 1.0) for data in edge_data.values())
                weights.append(total_weight)
            else:
                weights.append(1.0)
        
        # Weighted sampling
        if len(neighbors) <= num_samples:
            return neighbors, weights
        
        # Convert weights to probabilities
        weights = np.array(weights)
        probabilities = weights / np.sum(weights)
        
        # Sample without replacement
        sampled_indices = np.random.choice(
            len(neighbors), 
            size=num_samples, 
            replace=False, 
            p=probabilities
        )
        
        sampled_neighbors = [neighbors[i] for i in sampled_indices]
        sampled_weights = [weights[i] for i in sampled_indices]
        
        return sampled_neighbors, sampled_weights

class PinSageTrainer:
    """
    Trainer for PinSage model to learn user taste representations.
    """
    def __init__(self, model: PinSageModel, graph_builder: TasteGraphBuilder, 
                 learning_rate: float = 0.001, device: str = 'cpu'):
        self.model = model
        self.graph_builder = graph_builder
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Move model to device
        self.model.to(device)
        
    def train_epoch(self, user_ids: List[str], batch_size: int = 32) -> float:
        """
        Train for one epoch on user taste learning.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Shuffle users
        np.random.shuffle(user_ids)
        
        for i in range(0, len(user_ids), batch_size):
            batch_users = user_ids[i:i+batch_size]
            batch_loss = self._train_batch(batch_users)
            total_loss += batch_loss
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _train_batch(self, user_ids: List[str]) -> float:
        """
        Train on a batch of users.
        """
        batch_size = len(user_ids)
        if batch_size == 0:
            return 0.0
        
        # Prepare batch data
        node_features = []
        neighbor_features = []
        neighbor_weights = []
        
        for user_id in user_ids:
            node_id = f"user_{user_id}"
            
            if node_id not in self.graph_builder.graph:
                continue
            
            # Get node features
            node_feat = self.graph_builder.node_features.get(node_id, np.zeros(512))
            node_features.append(node_feat)
            
            # Sample neighbors
            neighbors, weights = self.graph_builder.sample_neighbors(node_id, 25)
            
            if not neighbors:
                # No neighbors - use zero padding
                neighbor_feat = np.zeros((25, 512))
                neighbor_weight = np.zeros(25)
            else:
                # Get neighbor features
                neighbor_feat = []
                neighbor_weight = []
                
                for neighbor, weight in zip(neighbors, weights):
                    feat = self.graph_builder.node_features.get(neighbor, np.zeros(512))
                    neighbor_feat.append(feat)
                    neighbor_weight.append(weight)
                
                # Pad to fixed size
                while len(neighbor_feat) < 25:
                    neighbor_feat.append(np.zeros(512))
                    neighbor_weight.append(0.0)
                
                neighbor_feat = np.array(neighbor_feat[:25])
                neighbor_weight = np.array(neighbor_weight[:25])
            
            neighbor_features.append(neighbor_feat)
            neighbor_weights.append(neighbor_weight)
        
        if not node_features:
            return 0.0
        
        # Convert to tensors
        node_features = torch.FloatTensor(np.array(node_features)).to(self.device)
        neighbor_features = torch.FloatTensor(np.array(neighbor_features)).to(self.device)
        neighbor_weights = torch.FloatTensor(np.array(neighbor_weights)).to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(node_features, neighbor_features, neighbor_weights)
        
        # Compute loss (reconstruction loss - predict current features)
        target = node_features
        loss = self.criterion(output, target)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_user_taste_vectors(self, user_ids: List[str]):
        """
        Update user taste vectors using trained model.
        """
        self.model.eval()
        
        with torch.no_grad():
            for user_id in user_ids:
                node_id = f"user_{user_id}"
                
                if node_id not in self.graph_builder.graph:
                    continue
                
                # Get current features
                node_feat = self.graph_builder.node_features.get(node_id, np.zeros(512))
                node_features = torch.FloatTensor(node_feat).unsqueeze(0).to(self.device)
                
                # Sample neighbors
                neighbors, weights = self.graph_builder.sample_neighbors(node_id, 25)
                
                if not neighbors:
                    neighbor_features = torch.zeros(1, 25, 512).to(self.device)
                    neighbor_weights = torch.zeros(1, 25).to(self.device)
                else:
                    neighbor_feat = []
                    neighbor_weight = []
                    
                    for neighbor, weight in zip(neighbors, weights):
                        feat = self.graph_builder.node_features.get(neighbor, np.zeros(512))
                        neighbor_feat.append(feat)
                        neighbor_weight.append(weight)
                    
                    # Pad to fixed size
                    while len(neighbor_feat) < 25:
                        neighbor_feat.append(np.zeros(512))
                        neighbor_weight.append(0.0)
                    
                    neighbor_features = torch.FloatTensor(np.array(neighbor_feat[:25])).unsqueeze(0).to(self.device)
                    neighbor_weights = torch.FloatTensor(np.array(neighbor_weight[:25])).unsqueeze(0).to(self.device)
                
                # Forward pass
                new_taste_vector = self.model(node_features, neighbor_features, neighbor_weights)
                
                # Update in database
                taste_vector = new_taste_vector.cpu().numpy().squeeze()
                self.graph_builder.session.update_user_taste_vector(user_id, taste_vector)
                
                # Update in-memory features
                self.graph_builder.node_features[node_id] = taste_vector

class RealTimeTasteUpdater:
    """
    Real-time taste vector updater for immediate user preference changes.
    """
    def __init__(self, session, vectorizer, index_adapter=None, learning_rate: float = 0.1):
        self.session = session
        self.vectorizer = vectorizer
        self.index_adapter = index_adapter
        self.learning_rate = learning_rate
        
    def update_user_taste_on_interaction(self, user_id: str, image_id: str, 
                                       event: str, weight: float = 1.0):
        """
        Update user taste vector immediately after an interaction.
        """
        # Get current user taste vector
        current_taste = self.session.get_user_taste_vector(user_id)
        
        if current_taste is None:
            # Initialize user taste vector
            current_taste = np.random.normal(0, 0.1, 512)
        
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
        
        # Compute interaction strength
        interaction_strength = self._get_interaction_strength(event, weight)
        
        # Update taste vector using simple weighted average
        # This is a simplified real-time update - PinSage would be more sophisticated
        alpha = self.learning_rate * interaction_strength
        
        new_taste = (1 - alpha) * current_taste + alpha * image_vector
        
        # Normalize
        new_taste = new_taste / (np.linalg.norm(new_taste) + 1e-8)
        
        # Update in database
        self.session.update_user_taste_vector(user_id, new_taste)
        
        # Update topic affinities
        self._update_topic_affinities(user_id, image_id, event, weight)
        
        logger.info(f"Updated taste vector for user {user_id} after {event}")
    
    def _get_interaction_strength(self, event: str, weight: float) -> float:
        """Convert interaction to learning strength"""
        event_strengths = {
            'save': 0.3,
            'like': 0.2,
            'click': 0.1,
            'view': 0.05
        }
        return event_strengths.get(event, 0.1) * weight
    
    def _update_topic_affinities(self, user_id: str, image_id: str, event: str, weight: float):
        """Update user's topic affinities based on interaction"""
        # Get topics associated with this image
        pin_topics = self.session.session.query(PinTopic).filter(
            PinTopic.pin_id == image_id
        ).all()
        
        # Update affinity for each topic
        affinity_delta = self._get_interaction_strength(event, weight)
        
        for pin_topic in pin_topics:
            self.session.update_user_topic_affinity(
                user_id, pin_topic.topic_id, affinity_delta, event,
                vectorizer=self.vectorizer, index_adapter=self.index_adapter
            )

def initialize_taste_graph(session, vectorizer, index_adapter=None):
    """
    Initialize the taste graph system.
    """
    logger.info("Initializing taste graph system...")
    
    # Create graph builder
    graph_builder = TasteGraphBuilder(session, vectorizer, index_adapter)
    
    # Build initial graph
    graph_builder.build_graph()
    
    # Initialize PinSage model
    model = PinSageModel(
        input_dim=512,
        hidden_dims=[256, 128],
        output_dim=512,
        num_samples=25
    )
    
    # Create trainer
    trainer = PinSageTrainer(model, graph_builder)
    
    # Create real-time updater
    realtime_updater = RealTimeTasteUpdater(session, vectorizer, index_adapter)
    
    return {
        'graph_builder': graph_builder,
        'model': model,
        'trainer': trainer,
        'realtime_updater': realtime_updater
    }
