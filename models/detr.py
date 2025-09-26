# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes for anatomical organ/lesion forecasting.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized)

from .backbone import PointBackbone
from .matcher import build_matcher
from .transformer import build_transformer

class PrototypeClassifier(nn.Module):
    """
    Prototype-based classifier using cosine similarity with learnable temperature and bias.
    
    Maps input features to prototype space, then computes cosine similarities with class prototypes.
    Forms logits as τ · cos + b, where τ is learnable temperature and b is per-class bias.
    """
    
    def __init__(self, input_dim: int, prototype_dim: int, num_classes: int, prototype_embeddings: nn.Embedding, num_layers: int = 2):
        """
        Initialize prototype classifier.
        
        Args:
            input_dim: Dimension of input features (hidden_dim)
            prototype_dim: Dimension of prototype space (subclass_dim)
            num_classes: Number of classes (num_subclasses)
            prototype_embeddings: Pre-trained prototype embeddings from backbone
        """
        super().__init__()
        self.num_classes = num_classes
        self.prototype_dim = prototype_dim
        
        # Map input to prototype space
        self.feature_mapper = MLP(input_dim, 2*input_dim, prototype_dim, num_layers)
        
        # Use the backbone's prototype embeddings (shared parameters)
        self.prototype_embeddings = prototype_embeddings
        
        # Temperature parameter (global scale)
        self.temperature = 3 * math.sqrt(prototype_dim)
        
        # Learnable per-class bias vector
        self.bias = nn.Parameter(torch.zeros(num_classes))
        
    
    def forward(self, x):
        """
        Forward pass through prototype classifier.
        
        Args:
            x: Input features of shape [..., input_dim]
            
        Returns:
            Logits of shape [..., num_classes]
        """
        # Map input to prototype space
        mapped_features = self.feature_mapper(x)  # [..., prototype_dim]
        
        # L2-normalize mapped features
        mapped_features = F.normalize(mapped_features, p=2, dim=-1)
        
        # Get prototype embeddings and L2-normalize them
        prototypes = self.prototype_embeddings.weight  # [num_classes, prototype_dim]
        prototypes = F.normalize(prototypes, p=2, dim=-1)
        
        # Compute cosine similarities
        # mapped_features: [..., prototype_dim]
        # prototypes: [num_classes, prototype_dim]
        cos_similarities = torch.matmul(mapped_features, prototypes.T)  # [..., num_classes]
        
        # Form logits: τ · cos + b
        logits = self.temperature * cos_similarities + self.bias
        
        return logits


class DETR(nn.Module):
    """ DETR module for anatomical organ/lesion forecasting with hierarchical classification """
    def __init__(self, backbone, transformer, num_superclasses, num_subclasses, num_queries, original_feature_size, aux_loss=False, use_film=False):
        """ Initializes the model.
        Parameters:
            backbone: PointBackbone module for point cloud processing
            transformer: torch module of the transformer architecture
            num_superclasses: number of superclass categories
            num_subclasses: number of subclass categories
            num_queries: number of object queries (max objects to predict)
            original_feature_size: size of original features per point
            aux_loss: True if auxiliary decoding losses are to be used
            use_film: Whether to use FiLM conditioning
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_superclasses = num_superclasses
        self.num_subclasses = num_subclasses
        self.use_film = use_film
        hidden_dim = transformer.d_model
        
        # Hierarchical classification heads
        self.superclass_embed = MLP(hidden_dim, 2*hidden_dim, num_superclasses + 1, num_layers=3) # include no-object class
        self.subclass_embed = PrototypeClassifier(hidden_dim, backbone.subclass_dim, num_subclasses, backbone.subclass_embeddings, num_layers=2)
        
        # Radiomics prediction head (includes 3D coordinates + radiomics features)
        # Note: original_feature_size now includes coords + radiomics + superclass + subclass
        # We predict coords + radiomics (excluding both labels)
        self.radiomics_embed = MLP(hidden_dim, 2*hidden_dim, original_feature_size - 2, num_layers=3) # exclude both labels
        
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Identity projection since PointBackbone already outputs hidden_dim
        self.input_proj = nn.Identity()
        
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_superclass": The predicted superclass logits for all queries.
                                        Shape= [batch_size x num_queries x (num_superclasses + 1)]
                - "pred_subclass": The predicted subclass logits for all queries.
                                    Shape= [batch_size x num_queries x num_subclasses]  
               - "pred_coordinates": The 3D coordinates for all queries.
                                     Shape= [batch_size x num_queries x 3]
               - "pred_radiomics": The radiomics features for all queries.
                                   Shape= [batch_size x num_queries x radiomics_dim]
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        features, film_conditioning = self.backbone(samples)

        # For point clouds, features is directly a NestedTensor
        src = features.tensors  # [batch, max_tokens, hidden_dim]
        mask = features.mask    # [batch, max_tokens]
        
        # Transformer expects: [seq_len, batch, hidden_dim]
        src = src.transpose(0, 1)  # [max_tokens, batch, hidden_dim]
        
        # For point clouds, position encoding is None
        pos = None
        
        # Pass FiLM conditioning if available
        if self.use_film and film_conditioning is not None:
            hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos, 
                                film_conditioning=film_conditioning)[0]
        else:
            hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)[0]

        outputs_superclass = self.superclass_embed(hs)
        outputs_subclass = self.subclass_embed(hs)
        outputs_full_radiomics = self.radiomics_embed(hs)  # Full prediction including coords
        
        # Split the radiomics output: first 3 are coordinates, rest are radiomics features
        outputs_coordinates = outputs_full_radiomics[..., :3]  # [layers, batch, queries, 3]
        outputs_radiomics = outputs_full_radiomics[..., 3:]    # [layers, batch, queries, remaining_features]
        
        out = {
            'pred_superclass': outputs_superclass[-1], 
            'pred_subclass': outputs_subclass[-1],
            'pred_coordinates': outputs_coordinates[-1],
            'pred_radiomics': outputs_radiomics[-1]
        }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_superclass, outputs_subclass, outputs_coordinates, outputs_radiomics)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_superclass, outputs_subclass, outputs_coordinates, outputs_radiomics):
        # Auxiliary losses for intermediate decoder layers
        return [{'pred_superclass': a, 'pred_subclass': b, 'pred_coordinates': c, 'pred_radiomics': d}
                for a, b, c, d in zip(outputs_superclass[:-1], outputs_subclass[:-1], outputs_coordinates[:-1], outputs_radiomics[:-1])]

    def update_ema_embeddings(self):
        """
        Update EMA embeddings in the backbone.
        Should be called after each training step to update the input embeddings 
        with exponential moving average of the output prototype embeddings.
        """
        self.backbone.update_ema_embeddings()


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR with hierarchical classification.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_superclasses, num_subclasses, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_superclasses: number of superclass categories, omitting the special no-object category
            num_subclasses: number of subclass categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_superclasses = num_superclasses
        self.num_subclasses = num_subclasses
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        
        # Empty weights for superclass
        empty_weight_super = torch.ones(self.num_superclasses + 1)
        empty_weight_super[-1] = self.eos_coef
        self.register_buffer('empty_weight_super', empty_weight_super)
        
        # Subclass weights with higher weight for specific minority classes # TODO: put as main.py argument
        empty_weight_sub = torch.ones(self.num_subclasses)
        high_weight_subclass_ids = [22, 24, 25, 28, 29, 31, 32, 33, 35, 36, 37, 45, 46, 50, 53, 54, 56, 57, 59, 60, 61, 62, 63, 64, 68, 69, 71, 72, 76, 78, 80, 82, 84, 87, 89, 90]
        for subclass_id in high_weight_subclass_ids:
            if subclass_id < self.num_subclasses:  # Safety check
                empty_weight_sub[subclass_id] = 5.0 # based on rough class distribution
        self.register_buffer('empty_weight_sub', empty_weight_sub)

    def loss_superclass_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Superclass classification loss (NLL)
        targets dicts must contain the key "superclass" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_superclass' in outputs
        src_logits = outputs['pred_superclass']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["superclass"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_superclasses,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight_super)
        losses = {'loss_superclass_ce': loss_ce}

        if log:
            losses['superclass_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_subclass_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Subclass classification loss (NLL) - for all queries matched to ground truth objects
        targets dicts must contain the key "subclass" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_subclass' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_subclass_logits = outputs['pred_subclass'][idx]
        
        target_subclasses_o = torch.cat([t["subclass"][J] for t, (_, J) in zip(targets, indices)])
        
        
        loss_ce = F.cross_entropy(src_subclass_logits, target_subclasses_o, weight=self.empty_weight_sub)
        losses = {'loss_subclass_ce': loss_ce}
        
        if log:
            losses['subclass_error'] = 100 - accuracy(src_subclass_logits, target_subclasses_o)[0]
                
        return losses



    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_superclass']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["superclass"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_coordinates(self, outputs, targets, indices, num_boxes):
        """Compute the 3D coordinates prediction loss (MSE)
           targets dicts must contain the key "coordinates" containing a tensor of dim [nb_target_boxes, 3]
        """
        assert 'pred_coordinates' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_coordinates = outputs['pred_coordinates'][idx]
        target_coordinates = torch.cat([t['coordinates'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # MSE loss for 3D coordinates
        loss_coordinates = F.mse_loss(src_coordinates, target_coordinates, reduction='none')

        losses = {}
        losses['loss_coordinates'] = loss_coordinates.sum() / num_boxes
        return losses

    def loss_radiomics(self, outputs, targets, indices, num_boxes):
        """Compute the radiomics prediction loss (MSE)
           targets dicts must contain the key "radiomics" containing a tensor of dim [nb_target_boxes, feature_size]
        """
        assert 'pred_radiomics' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_radiomics = outputs['pred_radiomics'][idx]
        target_radiomics = torch.cat([t['radiomics'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_radiomics = F.mse_loss(src_radiomics, target_radiomics, reduction='none')

        losses = {}
        losses['loss_radiomics'] = loss_radiomics.sum() / num_boxes
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'superclass_labels': self.loss_superclass_labels,
            'subclass_labels': self.loss_subclass_labels,
            'cardinality': self.loss_cardinality,
            'coordinates': self.loss_coordinates,
            'radiomics': self.loss_radiomics,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["superclass"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'superclass_labels' or loss == 'subclass_labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def zero_biases(self):
        """Zero out biases in the MLP layers."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear) and layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.leaky_relu(layer(x), 0.1) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    """
    Build DETR model for anatomical organ/lesion forecasting with hierarchical classification.
    """
    device = torch.device(args.device)

    # Build point cloud backbone with hierarchical label support
    backbone = PointBackbone(
        num_features=args.original_feature_size,
        num_subclasses=args.num_subclasses,
        subclass_dim=getattr(args, 'subclass_dim', 64),
        hidden_dim=args.hidden_dim,
        ema_momentum=getattr(args, 'ema_momentum', 0.995),
        use_film=getattr(args, 'use_film', False)
    )

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_superclasses=args.num_superclasses,
        num_subclasses=args.num_subclasses,
        num_queries=args.num_queries,
        original_feature_size=args.original_feature_size,
        aux_loss=args.aux_loss,
        use_film=getattr(args, 'use_film', False),
    )
    
    matcher = build_matcher(args)
    
    # Weight dictionary for hierarchical losses
    weight_dict = {
        'loss_superclass_ce': getattr(args, 'superclass_loss_coef', 2),
        'loss_subclass_ce': getattr(args, 'subclass_loss_coef', 2),
        'loss_coordinates': getattr(args, 'coordinates_loss_coef', 5),
        'loss_radiomics': getattr(args, 'radiomics_loss_coef', 5)
    }
    
    # Add auxiliary loss weights if needed
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # Losses for hierarchical point cloud forecasting
    losses = ['superclass_labels', 'subclass_labels', 'coordinates', 'radiomics', 'cardinality']
    
    criterion = SetCriterion(args.num_superclasses, args.num_subclasses, matcher=matcher, 
                             weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)

    return model, criterion
