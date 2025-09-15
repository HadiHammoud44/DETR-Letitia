# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor


class PositionEmbeddingZero(nn.Module):
    """
    Zero positional embedding for point clouds.
    
    For point cloud data, spatial relationships are already encoded in the 3D coordinates
    of each point, so we don't need additional positional encodings. This class returns
    zero embeddings to maintain compatibility with the transformer architecture.
    """
    def __init__(self):
        """
        Initialize zero position embedding.
        No parameters needed since we return zeros based on input dimensions.
        """
        super().__init__()

    def forward(self, tensor_list: NestedTensor):
        """
        Forward pass returning zero positional embeddings.
        
        Args:
            tensor_list: NestedTensor containing:
                - tensors: [Batch, max_tokens, hidden_dim] - embedded point clouds
                - mask: [Batch, max_tokens] - padding mask
        
        Returns:
            Zero tensor of shape [Batch, hidden_dim, max_tokens] for compatibility
        """
        x = tensor_list.tensors  # [Batch, max_tokens, hidden_dim]
        batch_size, max_tokens, hidden_dim = x.shape
        
        # Return zeros with shape [Batch, hidden_dim, max_tokens] to match expected format
        pos = torch.zeros(batch_size, hidden_dim, max_tokens, device=x.device, dtype=x.dtype)
        return pos


def build_position_encoding(args):
    """
    Build position encoding for point cloud processing.
    
    Since point clouds already contain 3D spatial information in their coordinates,
    we use zero positional embeddings to maintain transformer compatibility
    without adding redundant positional information.
    
    Args:
        args: Arguments containing model configuration
        
    Returns:
        PositionEmbeddingZero instance
    """
    position_embedding = PositionEmbeddingZero()
    return position_embedding
