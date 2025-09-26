# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_superclass: float = 1, cost_subclass: float = 1, cost_coordinates: float = 1, cost_radiomics: float = 1):
        """Creates the matcher

        Params:
            cost_superclass: This is the relative weight of the superclass classification error in the matching cost
            cost_subclass: This is the relative weight of the subclass classification error in the matching cost
            cost_coordinates: This is the relative weight of the L2 error of the 3D coordinates in the matching cost
            cost_radiomics: This is the relative weight of the L2 error of the radiomics features in the matching cost
        """
        super().__init__()
        self.cost_superclass = cost_superclass
        self.cost_subclass = cost_subclass
        self.cost_coordinates = cost_coordinates
        self.cost_radiomics = cost_radiomics
        assert (cost_superclass != 0 or cost_subclass != 0 or cost_coordinates != 0 or cost_radiomics != 0), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_superclass": Tensor of dim [batch_size, num_queries, num_superclasses + 1] with the superclass and no_object logits
                 "pred_subclass": Tensor of dim [batch_size, num_queries, num_subclasses] with the subclass logits
                 "pred_coordinates": Tensor of dim [batch_size, num_queries, 3] with the predicted 3D coordinates
                 "pred_radiomics": Tensor of dim [batch_size, num_queries, num_radiomics] with the predicted radiomics features

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "superclass": Tensor of dim [num_target_points] containing the superclass labels
                 "subclass": Tensor of dim [num_target_points] containing the subclass labels
                 "coordinates": Tensor of dim [num_target_points, 3] containing the target 3D coordinates
                 "radiomics": Tensor of dim [num_target_points, num_radiomics] containing the target radiomics features

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_points)
        """
        bs, num_queries = outputs["pred_superclass"].shape[:2]
        
        out_prob_super = outputs["pred_superclass"].flatten(0, 1).softmax(-1)
        out_prob_sub = outputs["pred_subclass"].flatten(0, 1).softmax(-1)
        out_coordinates = outputs["pred_coordinates"].flatten(0, 1)  # [batch_size * num_queries, 3]
        out_radiomics = outputs["pred_radiomics"].flatten(0, 1)  # [batch_size * num_queries, num_radiomics]

        tgt_superclass_ids = torch.cat([v["superclass"] for v in targets])
        tgt_subclass_ids = torch.cat([v["subclass"] for v in targets])
        tgt_coordinates = torch.cat([v["coordinates"] for v in targets])
        tgt_radiomics = torch.cat([v["radiomics"] for v in targets])

        # Compute the superclass classification cost
        cost_superclass = -out_prob_super[:, tgt_superclass_ids]

        # Compute the subclass classification cost
        cost_subclass = -out_prob_sub[:, tgt_subclass_ids]

        # Compute the L2 cost between 3D coordinates
        cost_coordinates = torch.cdist(out_coordinates, tgt_coordinates, p=2)

        # Compute the L2 cost between radiomics features
        cost_radiomics = torch.cdist(out_radiomics, tgt_radiomics, p=2)

        # Debug: Check for NaN/inf values
        if torch.isnan(cost_superclass).any():
            print(f"WARNING: NaN values found in cost_superclass")
            print(f"out_prob_super has NaN: {torch.isnan(out_prob_super).any()}")
            print(f"tgt_superclass_ids: {tgt_superclass_ids}")
        
        if torch.is_tensor(cost_subclass) and torch.isnan(cost_subclass).any():
            print(f"WARNING: NaN values found in cost_subclass")
        
        if torch.isnan(cost_coordinates).any():
            print(f"WARNING: NaN values found in cost_coordinates")
            print(f"out_coordinates has NaN: {torch.isnan(out_coordinates).any()}")
            print(f"tgt_coordinates has NaN: {torch.isnan(tgt_coordinates).any()}")
        
        if torch.isnan(cost_radiomics).any():
            print(f"WARNING: NaN values found in cost_radiomics")
            print(f"out_radiomics has NaN: {torch.isnan(out_radiomics).any()}")
            print(f"tgt_radiomics has NaN: {torch.isnan(tgt_radiomics).any()}")

        # Final cost matrix
        C = (self.cost_superclass * cost_superclass + 
             self.cost_subclass * cost_subclass + 
             self.cost_coordinates * cost_coordinates + 
             self.cost_radiomics * cost_radiomics)
        
        # Debug: Check final cost matrix
        if torch.isnan(C).any() or torch.isinf(C).any():
            print(f"WARNING: Invalid values in final cost matrix C")
            print(f"C has NaN: {torch.isnan(C).any()}")
            print(f"C has inf: {torch.isinf(C).any()}")

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["superclass"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
def build_matcher(args):
    return HungarianMatcher(
        cost_superclass=getattr(args, 'set_cost_superclass', 1),
        cost_subclass=getattr(args, 'set_cost_subclass', 1), 
        cost_coordinates=getattr(args, 'set_cost_coordinates', 1),
        cost_radiomics=getattr(args, 'set_cost_radiomics', 1)
    )
