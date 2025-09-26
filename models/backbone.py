# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules for point cloud processing.
"""
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

from util.misc import NestedTensor


class PointEmbed(nn.Module):
    """
    Point cloud embedding module for anatomical organs/lesions with learnable subclass embeddings.
    
    Transforms point cloud features from [coordinates, radiomics, superclass, subclass] format
    to [coordinates, radiomics, superclass, subclass_embedding] and then to hidden representation.
    
    Input: [Batch, max_tokens, num_features] - point clouds with radiomics features
    Output: [Batch, max_tokens, out_features] - embedded representations
    """
    
    def __init__(self, num_features: int, num_subclasses: int, subclass_dim: int = 64, hidden_dim: int = 256, ema_momentum: float = 0.995, use_film: bool = False):
        """
        Initialize point embedding module with learnable subclass embeddings.
        
        Args:
            num_features: Number of input features per point (3D coords + radiomics + superclass + subclass)
            num_subclasses: Number of unique subclass labels for embedding lookup
            subclass_dim: Dimension of learnable subclass embeddings (default: 64)
            hidden_dim: Output embedding dimension (default: 256)
            ema_momentum: Momentum for EMA updates (default: 0.995)
            use_film: Whether to use FiLM conditioning (default: False)
        """
        super().__init__()
        self.num_channels = hidden_dim  # For compatibility with downstream code
        self.num_subclasses = num_subclasses
        self.subclass_dim = subclass_dim
        self.ema_momentum = ema_momentum
        self.use_film = use_film
        
        # Learnable subclass embeddings for output classification, initialized with orthogonal initialization
        self.subclass_embeddings = nn.Embedding(num_subclasses, subclass_dim)
        
        # EMA embeddings for input processing (detached from gradients)
        # These will be updated with EMA from the output prototypes
        self.ema_embeddings = nn.Embedding(num_subclasses, subclass_dim)
        self.ema_embeddings.requires_grad_(False)  # No gradients for EMA embeddings
        
        if use_film:
            # For FiLM mode: only process 'rest' features (without subclass embeddings)
            rest_dim = num_features - 1  # [coords(3), radiomics(N), superclass(1)]
            self.embed = nn.Sequential(
                nn.Linear(rest_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            # Original mode: concatenate subclass embeddings with rest
            # Calculate intermediate feature dimension after replacing subclass with embedding
            # Original: [coords(3), radiomics(N), superclass(1), subclass(1)]
            # Intermediate: [coords(3), radiomics(N), superclass(1), subclass_embedding(subclass_dim)]
            intermediate_dim = num_features - 1 + subclass_dim 
            
            # Embedding transformation: intermediate → hidden → output
            self.embed = nn.Sequential(
                nn.Linear(intermediate_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights and zero out biases.
        """
        # Initialize subclass embeddings with orthogonal initialization
        with torch.no_grad():
            M = torch.empty(self.subclass_dim, self.num_subclasses, 
                          device=self.subclass_embeddings.weight.device, 
                          dtype=self.subclass_embeddings.weight.dtype)
            torch.nn.init.orthogonal_(M)  # columns orthonormal since dim ≥ Classes
            self.subclass_embeddings.weight.copy_(M.T)
            
            # Initialize EMA embeddings with the same values as subclass embeddings
            self.ema_embeddings.weight.copy_(self.subclass_embeddings.weight.data)
        
        # Zero out biases in embedding layers
        for layer in self.embed:
            if isinstance(layer, nn.Linear) and layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, tensor_list: NestedTensor) -> Tuple[NestedTensor, None]:
        """
        Forward pass through point embedding with subclass embedding replacement.
        
        Args:
            tensor_list: NestedTensor containing:
                - tensors: [Batch, max_tokens, num_features] - padded point clouds
                  Features: [x, y, z, radiomics..., superclass_label, subclass_label]
                - mask: [Batch, max_tokens] - padding mask (True = padded)
        
        Returns:
            Tuple of (embedded_features, ema_embeddings) where:
            - embedded_features: NestedTensor with embedded point representations
            - ema_embeddings: For FiLM mode, returns conditioning embeddings; otherwise None
        """
        # Extract point cloud data and mask
        point_data = tensor_list.tensors  # [Batch, max_tokens, num_features]
        mask = tensor_list.mask          # [Batch, max_tokens]
        
        # Split the input features
        subclass_labels = point_data[:, :, -1].long()        # [Batch, max_tokens]
        superclass_labels = point_data[:, :, -2]             # [Batch, max_tokens] - binary feature
        rest = point_data[:, :, :-1]                          # [Batch, max_tokens, num_features - 1]
        
        # Clamp subclass labels to valid range to handle potential padding values
        subclass_labels = torch.clamp(subclass_labels, 0, self.num_subclasses - 1)
        
        # Get EMA subclass embeddings (detached from gradients)
        subclass_embeds = self.ema_embeddings(subclass_labels)  # [Batch, max_tokens, subclass_dim]
        
        # Normalize the subclass embeddings
        subclass_embeds = F.normalize(subclass_embeds, p=2, dim=-1)  # L2 normalization
        
        if self.use_film:
            # FiLM mode: process only 'rest' features, return embeddings separately
            embedded = self.embed(rest)  # [Batch, max_tokens, hidden_dim]
            
            # Transform superclass binary feature to {-1, 1} and add to conditioning
            superclass_binary = 2.0 * superclass_labels - 1.0  # Transform {0,1} -> {-1,1}
            superclass_binary = superclass_binary.unsqueeze(-1)  # [Batch, max_tokens, 1]
            
            # Concatenate subclass embeddings with binary superclass feature
            film_conditioning = torch.cat([
                subclass_embeds,        # [Batch, max_tokens, subclass_dim]
                superclass_binary       # [Batch, max_tokens, 1]
            ], dim=-1)  # [Batch, max_tokens, subclass_dim + 1]
            
            # Return as NestedTensor with original mask, and film_conditioning
            return NestedTensor(embedded, mask), film_conditioning
        else:
            # Original mode: concatenate subclass embeddings with rest features
            # Reconstruct intermediate feature vector
            intermediate_features = torch.cat([
                rest,                                  # [Batch, max_tokens, num_features - 1]
                subclass_embeds                       # [Batch, max_tokens, subclass_dim]
            ], dim=-1)  # [Batch, max_tokens, intermediate_dim]
            
            # Apply embedding transformation
            embedded = self.embed(intermediate_features)  # [Batch, max_tokens, hidden_dim]
            
            # Return as NestedTensor with original mask, and None for position encoding
            return NestedTensor(embedded, mask), None

    def update_ema_embeddings(self):
        """
        Update EMA embeddings with exponential moving average from the current subclass embeddings.
        Should be called after each training step.
        """
        if not self.training:
            return
            
        with torch.no_grad():
            # EMA update: ema = momentum * ema + (1 - momentum) * current
            self.ema_embeddings.weight.mul_(self.ema_momentum).add_(
                self.subclass_embeddings.weight.detach(), alpha=1 - self.ema_momentum
            )


class PointBackbone(nn.Module):
    """
    Point cloud backbone that wraps PointEmbed for DETR integration.
    """
    
    def __init__(self, num_features: int, num_subclasses: int, subclass_dim: int = 64, hidden_dim: int = 256, ema_momentum: float = 0.995, use_film: bool = False):
        """
        Initialize point cloud backbone.
        
        Args:
            num_features: Number of input features per point
            num_subclasses: Number of unique subclass labels
            subclass_dim: Dimension of learnable subclass embeddings
            hidden_dim: Embedding dimension
            ema_momentum: Momentum for EMA updates (default: 0.995)
            use_film: Whether to use FiLM conditioning (default: False)
        """
        super().__init__()
        self.point_embed = PointEmbed(num_features, num_subclasses, subclass_dim, hidden_dim, ema_momentum, use_film)
        self.num_channels = hidden_dim
        self.subclass_dim = subclass_dim
        self.subclass_embeddings = self.point_embed.subclass_embeddings 
        self.use_film = use_film 
    
    def forward(self, tensor_list: NestedTensor) -> Tuple[NestedTensor, None]:
        """
        Forward pass through point backbone.
        
        Args:
            tensor_list: NestedTensor with point cloud data
            
        Returns:
            Tuple of (embedded_features, ema_embeddings_or_None)
            - For FiLM mode: (features, ema_embeddings)
            - For regular mode: (features, None)
        """
        return self.point_embed(tensor_list)

    def update_ema_embeddings(self):
        """
        Update EMA embeddings in the point embedding module.
        Should be called after each training step.
        """
        self.point_embed.update_ema_embeddings()

