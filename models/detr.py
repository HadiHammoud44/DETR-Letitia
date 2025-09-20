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


class DETR(nn.Module):
    """ DETR module for anatomical organ/lesion forecasting """
    def __init__(self, backbone, transformer, num_superclasses, num_queries, original_feature_size, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: PointBackbone module for point cloud processing
            transformer: torch module of the transformer architecture
            num_superclasses: number of superclass categories
            num_queries: number of object queries (max objects to predict)
            original_feature_size: size of original features per point
            aux_loss: True if auxiliary decoding losses are to be used
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_superclasses = num_superclasses
        hidden_dim = transformer.d_model
        
        # Superclass classification head only
        self.superclass_embed = MLP(hidden_dim, 2*hidden_dim, num_superclasses + 1, num_layers=3) # include no-object class
        
        # Radiomics prediction head (includes 3D coordinates + radiomics features)
        # We predict coords + radiomics (excluding empty_pt and superclass)
        self.radiomics_embed = MLP(hidden_dim, 2*hidden_dim, original_feature_size - 2, num_layers=3) # exclude empty_pt and superclass
        
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
               - "pred_coordinates": The 3D coordinates for all queries.
                                     Shape= [batch_size x num_queries x 3]
               - "pred_radiomics": The radiomics features for all queries.
                                   Shape= [batch_size x num_queries x radiomics_dim]
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        features, _ = self.backbone(samples)  # No conditioning/position encoding returned

        # For point clouds, features is directly a NestedTensor
        src = features.tensors  # [batch, max_tokens, hidden_dim]
        mask = features.mask    # [batch, max_tokens]
        
        # Transformer expects: [seq_len, batch, hidden_dim]
        src = src.transpose(0, 1)  # [max_tokens, batch, hidden_dim]
        
        # For point clouds, position encoding is None
        pos = None
        
        # Simplified transformer call
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos)[0]

        outputs_superclass = self.superclass_embed(hs)
        outputs_full_radiomics = self.radiomics_embed(hs)  # Full prediction including coords
        
        # Split the radiomics output: first 3 are coordinates, rest are radiomics features
        outputs_coordinates = outputs_full_radiomics[..., :3]  # [layers, batch, queries, 3]
        outputs_radiomics = outputs_full_radiomics[..., 3:]    # [layers, batch, queries, remaining_features]
        
        out = {
            'pred_superclass': outputs_superclass[-1], 
            'pred_coordinates': outputs_coordinates[-1],
            'pred_radiomics': outputs_radiomics[-1]
        }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_superclass, outputs_coordinates, outputs_radiomics)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_superclass, outputs_coordinates, outputs_radiomics):
        # Auxiliary losses for intermediate decoder layers
        return [{'pred_superclass': a, 'pred_coordinates': b, 'pred_radiomics': c}
                for a, b, c in zip(outputs_superclass[:-1], outputs_coordinates[:-1], outputs_radiomics[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and coordinates)
    """
    def __init__(self, num_superclasses, matcher, weight_dict, losses, superclass_coef):
        """ Create the criterion.
        Parameters:
            num_superclasses: number of superclass categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            superclass_coef: list of weights for each superclass to handle class imbalance including no-object.
        """
        super().__init__()
        self.num_superclasses = num_superclasses
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        
        self.register_buffer('superclass_coef', torch.tensor(superclass_coef))

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

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.superclass_coef)
        losses = {'loss_superclass_ce': loss_ce}

        if log:
            losses['superclass_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
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
        """Compute the selective radiomics prediction loss (MSE)
           targets dicts must contain the key "radiomics" containing a tensor of dim [nb_target_boxes, feature_size]
           and "empty_pt" containing a tensor of dim [nb_target_boxes] with binary flags
           
           Loss computation:
           - First half of radiomics: always computed
           - Second half of radiomics: only computed where empty_pt = 0
        """
        assert 'pred_radiomics' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_radiomics = outputs['pred_radiomics'][idx]
        target_radiomics = torch.cat([t['radiomics'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_empty_pt = torch.cat([t['empty_pt'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # Split radiomics into two halves
        radiomics_dim = target_radiomics.shape[-1]
        half_dim = radiomics_dim // 2
        
        # First half: always compute loss
        src_first_half = src_radiomics[..., :half_dim]
        target_first_half = target_radiomics[..., :half_dim]
        loss_first_half = F.mse_loss(src_first_half, target_first_half, reduction='none')
        
        # Second half: compute loss only where empty_pt = 0 (not empty)
        src_second_half = src_radiomics[..., half_dim:]
        target_second_half = target_radiomics[..., half_dim:]
        loss_second_half = F.mse_loss(src_second_half, target_second_half, reduction='none')
        
        # Create mask for non-empty points (empty_pt = 0 means valid data)
        non_empty_mask = (target_empty_pt == 0).float().unsqueeze(-1)  # [N, 1]
        
        # Apply mask to second half loss
        masked_loss_second_half = loss_second_half * non_empty_mask
        
        # Combine losses
        total_loss = loss_first_half.sum() + masked_loss_second_half.sum()
        
        # Normalize by total number of valid elements
        # First half: always all elements, Second half: only non-empty elements
        num_first_half_elements = loss_first_half.numel()
        num_second_half_elements = non_empty_mask.sum().item() * (radiomics_dim - half_dim)
        total_elements = num_first_half_elements + num_second_half_elements
        
        losses = {}
        losses['loss_radiomics'] = total_loss / total_elements
            
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'superclass_labels': self.loss_superclass_labels,
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
                    if loss == 'superclass_labels':
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
    Build DETR model for anatomical organ/lesion forecasting.
    """
    device = torch.device(args.device)

    # Build simplified point cloud backbone
    backbone = PointBackbone(
        num_features=args.original_feature_size,  # Total feature size
        hidden_dim=args.hidden_dim
    )

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_superclasses=args.num_superclasses,
        num_queries=args.num_queries,
        original_feature_size=args.original_feature_size,
        aux_loss=args.aux_loss
    )
    
    matcher = build_matcher(args)
    
    # Weight dictionary for simplified losses
    weight_dict = {
        'loss_superclass_ce': getattr(args, 'superclass_loss_coef', 2),
        'loss_coordinates': getattr(args, 'coordinates_loss_coef', 5),
        'loss_radiomics': getattr(args, 'radiomics_loss_coef', 5)
    }
    
    # Add auxiliary loss weights if needed
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # Losses for point cloud forecasting
    losses = ['superclass_labels', 'coordinates', 'radiomics', 'cardinality']
    
    criterion = SetCriterion(args.num_superclasses, matcher=matcher, 
                             weight_dict=weight_dict, losses=losses, superclass_coef=args.superclass_coef)
    criterion.to(device)

    return model, criterion
