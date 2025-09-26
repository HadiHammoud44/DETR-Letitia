# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    
    Applies element-wise affine transformation: (1 + γ(c)) ⊙ x + β(c)
    where γ and β are learned functions of conditioning input c.
    """
    
    def __init__(self, conditioning_dim: int, feature_dim: int):
        """
        Initialize FiLM layer.
        
        Args:
            conditioning_dim: Dimension of conditioning input (subclass_dim + 1)
            feature_dim: Dimension of features to modulate (d_model)
        """
        super().__init__()
        self.conditioning_dim = conditioning_dim
        self.feature_dim = feature_dim
        
        # Single linear layer to generate both gamma and beta parameters
        self.to_gamma_beta = nn.Linear(conditioning_dim, 2 * feature_dim)
        
        # Zero-initialize for identity transformation initially
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)
    
    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        """
        Apply FiLM conditioning to input features.
        
        Args:
            x: Input features [seq_len, batch, feature_dim]
            conditioning: Conditioning input [batch, max_tokens, conditioning_dim]
                         Contains subclass embeddings + binary superclass feature
            
        Returns:
            Modulated features [seq_len, batch, feature_dim]
        """
        # Generate gamma and beta parameters
        gb = self.to_gamma_beta(conditioning)  # [batch, max_tokens, 2*feature_dim]
        gamma, beta = gb.chunk(2, dim=-1)      # [batch, max_tokens, feature_dim] each
        
        # Transpose to match x dimensions: [max_tokens, batch, feature_dim]
        gamma = gamma.transpose(0, 1)
        beta = beta.transpose(0, 1)
        
        # Apply FiLM: (1 + γ) ⊙ x + β
        return (1.0 + gamma) * x + beta


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, use_film=False, 
                 subclass_dim=64):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,
                                                use_film, subclass_dim)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm,
                                         use_film, subclass_dim)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.use_film = use_film

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, film_conditioning=None):
        # Handle 1D point cloud sequences: src is [seq_len, batch, hidden_dim]
        # src already comes transposed from detr.py: [max_tokens, batch, hidden_dim]
        seq_len, bs, hidden_dim = src.shape
        
        # Handle positional embedding (None for point clouds)
        if pos_embed is not None:
            # If position embedding is provided, ensure correct shape
            if pos_embed.dim() == 3:  # [batch, hidden_dim, seq_len]
                pos_embed = pos_embed.permute(2, 0, 1)  # [seq_len, batch, hidden_dim]
            elif pos_embed.dim() == 2:  # [batch, seq_len] 
                pos_embed = pos_embed.transpose(0, 1).unsqueeze(-1).expand(seq_len, bs, hidden_dim)
        else:
            # For point clouds with zero positional encoding
            pos_embed = torch.zeros_like(src)
        
        # Prepare query embeddings: [num_queries, batch, hidden_dim]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        
        # Key padding mask: mask is [batch, seq_len], no need to flatten for 1D sequences
        # PyTorch expects key_padding_mask as [batch, seq_len] where True = ignore
        
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed, 
                             film_conditioning=film_conditioning)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        
        # Return: [num_decoder_layers, batch, num_queries, hidden_dim], memory
        return hs.transpose(1, 2), memory


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, use_film=False, subclass_dim=64):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.use_film = use_film
        
        # Create FiLM layers if needed (2 per encoder layer)
        if use_film:
            # Conditioning dimension is subclass_dim + 1 (for binary superclass feature)
            conditioning_dim = subclass_dim + 1
            self.film_layers = nn.ModuleList([
                FiLMLayer(conditioning_dim, encoder_layer.self_attn.embed_dim)
                for _ in range(2 * num_layers)  # 2 FiLM layers per encoder layer
            ])

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                film_conditioning: Optional[Tensor] = None):
        output = src

        for i, layer in enumerate(self.layers):
            if self.use_film and film_conditioning is not None:
                # Pass FiLM layers for this encoder layer (2 layers per encoder)
                layer_film_layers = [
                    self.film_layers[2 * i],      # For norm1
                    self.film_layers[2 * i + 1]   # For norm2
                ]
                output = layer(output, src_mask=mask,
                             src_key_padding_mask=src_key_padding_mask, pos=pos,
                             film_layers=layer_film_layers, 
                             film_conditioning=film_conditioning)
            else:
                output = layer(output, src_mask=mask,
                             src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, use_film=False, subclass_dim=64):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.use_film = use_film

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre_film(self, src,
                         src_mask: Optional[Tensor] = None,
                         src_key_padding_mask: Optional[Tensor] = None,
                         pos: Optional[Tensor] = None,
                         film_layers: Optional[List[FiLMLayer]] = None,
                         film_conditioning: Optional[Tensor] = None):
        """
        Forward pass with FiLM conditioning applied to LayerNorm outputs.
        
        Args:
            src: Input features [seq_len, batch, d_model]
            film_layers: List of 2 FiLM layers [film_norm1, film_norm2]
            film_conditioning: Conditioning input [batch, max_tokens, subclass_dim + 1]
                              Contains subclass embeddings + binary superclass feature
        """
        # Apply LayerNorm and FiLM conditioning
        src2 = self.norm1(src)
        if film_layers is not None and film_conditioning is not None:
            src2 = film_layers[0](src2, film_conditioning)  # Apply FiLM to norm1 output
        
        # Self-attention
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        
        # Apply LayerNorm and FiLM conditioning
        src2 = self.norm2(src)
        if film_layers is not None and film_conditioning is not None:
            src2 = film_layers[1](src2, film_conditioning)  # Apply FiLM to norm2 output
            
        # Feed-forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src
    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                film_layers: Optional[List[FiLMLayer]] = None,
                film_conditioning: Optional[Tensor] = None):
        # Route to appropriate forward method based on FiLM usage
        if self.use_film and film_layers is not None and film_conditioning is not None:
            return self.forward_pre_film(src, src_mask, src_key_padding_mask, pos, 
                                        film_layers, film_conditioning)
        
        # Regular forward pass
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=True if args.use_film else args.pre_norm,
        return_intermediate_dec=True,
        use_film=args.use_film,
        subclass_dim=args.subclass_dim,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
