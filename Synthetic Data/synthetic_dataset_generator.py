"""
Synthetic Dataset Generator - "Rule-Driven Points in a Box"

Creates triple-timepoint datasets (T0 → T1 → T2) with deterministic point-cloud dynamics
for DETR-based temporal forecasting model training.

Note: 'extras' was not used by the reported experiments (deleted after generation using truncate.py),
and hence the fetaure vector size was 10 [position (3), velocity (3), energy (1), instability (1), class_bit (1), subclass_id (1)]
"""

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Tuple, List, Dict
import random
from copy import deepcopy

import sys
sys.path.append('/mnt/letitia/scratch/students/hhammoud/detr')

class SyntheticDatasetGenerator:
    """
    Generates synthetic triple-timepoint point cloud datasets with deterministic rules.
    
    Features:
    - Organs (class 0): persist throughout timepoints, and has subclass-dependent effect zone and impact on lesions
    - Lesions (class 1): may disappear or produce an additional lesion based on energy and instability
    - 3D coords + 3D velocity + energy + instability + extras (fixed MLP features; optional) + class_bit + subclass_id
    - Deterministic rule-based temporal evolution
    """
    
    # Global Constants 
    BOX_LIMIT = 1.0 # not used for now
    DELTA_T = 1.0 # time step
    SIGMA_POS = 0.01 # positional noise stddev
    
    # Enhanced dataset parameters
    MAX_ORGANS = 20  # Maximum number of organs per patient
    MAX_LESIONS = 72  # Maximum number of lesions per patient
    MIN_ORGANS = 10  # Minimum number of organs per patient
    MIN_LESIONS = 24  # Minimum number of lesions per patient
    
    def __init__(self, seed: int = 42):
        """
        Initialize the dataset generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.set_random_seed(seed)
        
        # Initialize the fixed MLP for 'extras' computation 
        self.extras_mlp = self._create_extras_mlp()
        
        # Initialize subclass configurations
        self._setup_subclass_configurations()
        
    def set_random_seed(self, seed: int):
        """Set random seeds for all RNGs for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _create_extras_mlp(self) -> nn.Module:
        """
        Create the fixed MLP for computing extras features .

        Input:  [coords (3), E (1), κ (1)] = 5 dims
        Output: 59 dims (extras features)
        """
        mlp = nn.Sequential(
            nn.Linear(5, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 59)
        )

        def _init_he_leaky(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu', mode='fan_in')
                nn.init.zeros_(m.bias)

        mlp.apply(_init_he_leaky)

        # Freeze the weights
        for p in mlp.parameters():
            p.requires_grad = False
        mlp.eval()  # (not necessary, but good practice)

        return mlp
    
    def _setup_subclass_configurations(self):
        """
        Setup global subclass configurations for organs and lesions.
        These configurations are fixed across all patients for reproducibility.
        """
        # Calculate total number of possible subclasses
        organ_range = self.MAX_ORGANS
        lesion_range = self.MAX_LESIONS
        self.total_subclasses = organ_range + lesion_range
        
        # Organ subclass configurations
        n_organ_subclasses = organ_range
        organ_subclass_ids = list(range(n_organ_subclasses))
        
        # Split organs into two effect types
        random.shuffle(organ_subclass_ids)
        mid_point = len(organ_subclass_ids) // 2
        
        self.organ_instability_boosters = set(organ_subclass_ids[:mid_point])
        self.organ_gamma_reducers = set(organ_subclass_ids[mid_point:])
        
        # Organ subclass parameters
        self.organ_radii = {}
        self.organ_gamma_effects = {}  # For GAMMA_GAIN reduction
        self.organ_instability_effects = {}  # For instability boost
        
        for subclass_id in range(n_organ_subclasses):
            # Uniform(0.3, 0.7) for radii
            self.organ_radii[subclass_id] = random.uniform(0.3, 0.7)
            
            if subclass_id in self.organ_gamma_reducers:
                # Uniform(0.2, 0.5) for GAMMA_GAIN suppression
                self.organ_gamma_effects[subclass_id] = random.uniform(0.2, 0.5)
            
            if subclass_id in self.organ_instability_boosters:
                # Uniform(1.5, 2.0) for instability boost
                self.organ_instability_effects[subclass_id] = random.uniform(1.5, 2.0)
        
        # Lesion subclass configurations
        n_lesion_subclasses = lesion_range
        lesion_subclass_ids = list(range(n_organ_subclasses, n_organ_subclasses + n_lesion_subclasses))
        
        # Split lesions into initial and evolved
        random.shuffle(lesion_subclass_ids)
        mid_point = len(lesion_subclass_ids) // 2
        
        self.initial_lesion_subclasses = set(lesion_subclass_ids[:mid_point])
        self.evolved_lesion_subclasses = set(lesion_subclass_ids[mid_point:])
        
        # Create transition map (initial -> evolved)
        initial_list = list(self.initial_lesion_subclasses)
        evolved_list = list(self.evolved_lesion_subclasses)
        random.shuffle(evolved_list)
        
        self.subclass_transition_map = {}
        for i, initial_id in enumerate(initial_list):
            evolved_id = evolved_list[i % len(evolved_list)]
            self.subclass_transition_map[initial_id] = evolved_id
        
        # Lesion subclass parameters
        self.lesion_gamma_gain = {}
        self.lesion_delta_decay = {}
        self.lesion_energy_dup_threshold = {}
        self.lesion_instability_dup_threshold = {}
        self.lesion_energy_death_threshold = {}
        
        all_lesion_subclasses = list(self.initial_lesion_subclasses) + list(self.evolved_lesion_subclasses)
        for subclass_id in all_lesion_subclasses:
            self.lesion_gamma_gain[subclass_id] = random.uniform(8.0, 20.0)
            self.lesion_delta_decay[subclass_id] = random.uniform(0.1, 0.35)
            self.lesion_energy_dup_threshold[subclass_id] = random.uniform(0.5, 0.7)
            self.lesion_instability_dup_threshold[subclass_id] = random.uniform(0.4, 0.7)
            self.lesion_energy_death_threshold[subclass_id] = random.uniform(0.15, 0.25)
        
        # Lesion-lesion interaction: assign nutrient field centers
        initial_lesions = list(self.initial_lesion_subclasses)
        random.shuffle(initial_lesions)
        mid_point = len(initial_lesions) // 2
        
        self.c1_lesion_subclasses = set(initial_lesions[:mid_point])
        self.c2_lesion_subclasses = set(initial_lesions[mid_point:])
        
        # Evolved lesions inherit their parent's field center
        self.lesion_field_assignment = {}
        for initial_id in self.c1_lesion_subclasses:
            self.lesion_field_assignment[initial_id] = 'C1'
            if initial_id in self.subclass_transition_map:
                evolved_id = self.subclass_transition_map[initial_id]
                self.lesion_field_assignment[evolved_id] = 'C1'
        
        for initial_id in self.c2_lesion_subclasses:
            self.lesion_field_assignment[initial_id] = 'C2'
            if initial_id in self.subclass_transition_map:
                evolved_id = self.subclass_transition_map[initial_id]
                self.lesion_field_assignment[evolved_id] = 'C2'
    
    def nutrient_field(self, coords: torch.Tensor, lesion_coords: torch.Tensor, 
                      lesion_subclasses: torch.Tensor, target_subclass: int) -> torch.Tensor:
        """
        Compute nutrient field based on dynamic centers C1 and C2.
        
        Args:
            coords: Target point coordinates, shape [..., 3]
            lesion_coords: All lesion coordinates, shape [N_lesions, 3]
            lesion_subclasses: All lesion subclass IDs, shape [N_lesions]
            target_subclass: Subclass ID of the target lesion
            
        Returns:
            Nutrient field values, shape [...]
        """
        if len(lesion_coords) == 0:
            # Fallback to original field if no lesions
            norm_squared = torch.sum(coords ** 2, dim=-1)
            return torch.exp(-norm_squared / (0.5 ** 2))
        
        # Determine which field center affects this lesion
        field_assignment = self.lesion_field_assignment.get(target_subclass)
        
        # Handle scalar tensor case
        if lesion_subclasses.dim() == 0:
            lesion_subclasses = lesion_subclasses.unsqueeze(0)
        
        # Calculate C1 and C2 centroids
        c1_mask = torch.tensor([self.lesion_field_assignment.get(int(sub_id)) == 'C1' 
                               for sub_id in lesion_subclasses], dtype=torch.bool)
        c2_mask = ~c1_mask
        
        if field_assignment == 'C1' and c1_mask.any():
            center = lesion_coords[c1_mask].mean(dim=0)
        elif field_assignment == 'C2' and c2_mask.any():
            center = lesion_coords[c2_mask].mean(dim=0)
        else:
            # Fallback to origin if no appropriate lesions found
            center = torch.zeros(3)
        
        # Compute field based on distance to center
        if coords.dim() == 1:
            distance_sq = torch.sum((coords - center) ** 2)
        else:
            distance_sq = torch.sum((coords - center) ** 2, dim=-1)
        
        return torch.exp(-distance_sq / (0.5 ** 2))
    
    def compute_extras(self, coords: torch.Tensor, energy: torch.Tensor, 
                      instability: torch.Tensor) -> torch.Tensor:
        """
        Compute the 59-dimensional extras features using the fixed MLP 
        (which will follow the velocity, energy, and instability to form a 64-D feature vector)
        
        Args:
            coords: Point coordinates, shape [N, 3]
            energy: Energy values, shape [N, 1]
            instability: Instability values, shape [N, 1]
            
        Returns:
            Extras features, shape [N, 59]
        """
        # Combine inputs: [coords, E, κ]
        mlp_input = torch.cat([coords, energy, instability], dim=-1)  # [N, 5]
        
        # Apply fixed MLP
        with torch.no_grad():
            extras = self.extras_mlp(mlp_input)  # [N, 59]
            
        # Add small noise
        # noise = torch.normal(0, 0.001, size=extras.shape)
        return extras #+ noise
    
    def initialize_organs(self, n_organs: int) -> Dict[str, torch.Tensor]:
        """
        Initialize organ points (class_bit = 0) with subclasses.
        
        Args:
            n_organs: Number of organs to create
            
        Returns:
            Dictionary with organ properties
        """
        # Coordinates: Uniform(-1, 1) in each dimension
        coords = torch.rand(n_organs, 3) * 2 - 1  # Uniform[-1, 1]
        
        # Velocity: Normal(0, 0.05)
        velocity = torch.normal(0, 0.05, size=(n_organs, 3))
        
        # Energy and instability: unused (set to zeros)
        energy = torch.zeros(n_organs, 1)
        instability = torch.zeros(n_organs, 1)
        
        # Assign unique subclass IDs from available organ subclasses
        organ_range = self.MAX_ORGANS
        available_subclasses = list(range(organ_range))
        if n_organs > len(available_subclasses):
            raise ValueError(f"Cannot create {n_organs} organs with only {len(available_subclasses)} unique subclasses")
        
        selected_subclasses = random.sample(available_subclasses, n_organs)
        subclass_id = torch.tensor(selected_subclasses, dtype=torch.float32).unsqueeze(1)
        
        # Compute extras using MLP
        extras = self.compute_extras(coords, energy, instability)
        
        # Class bit: 0 for organs
        class_bit = torch.zeros(n_organs, 1)
        
        return {
            'coords': coords,
            'velocity': velocity,
            'energy': energy,
            'instability': instability,
            'extras': extras,
            'class_bit': class_bit,
            'subclass_id': subclass_id
        }
    
    def initialize_lesions(self, n_lesions: int) -> Dict[str, torch.Tensor]:
        """
        Initialize lesion points (class_bit = 1) with subclasses.
        
        Args:
            n_lesions: Number of lesions to create
            
        Returns:
            Dictionary with lesion properties
        """
        # Coordinates: Uniform(-1, 1) in each dimension
        coords = torch.rand(n_lesions, 3) * 2 - 1  # Uniform[-1, 1]

        # Velocity: Normal(0, 0.3)
        velocity = torch.normal(0, 0.3, size=(n_lesions, 3))
        
        # Energy: Uniform(0.4, 0.7) - initial lesion energy
        energy = torch.rand(n_lesions, 1) * 0.3 + 0.4

        # Instability: Uniform(0.4, 0.7) - initial lesion instability
        instability = torch.rand(n_lesions, 1) * 0.3 + 0.4

        # Assign unique subclass IDs from available INITIAL lesion subclasses only
        available_subclasses = list(self.initial_lesion_subclasses)
        if n_lesions > len(available_subclasses):
            raise ValueError(f"Cannot create {n_lesions} initial lesions with only {len(available_subclasses)} unique initial subclasses")
        
        selected_subclasses = random.sample(available_subclasses, n_lesions)
        subclass_id = torch.tensor(selected_subclasses, dtype=torch.float32).unsqueeze(1)

        # Compute extras using MLP
        extras = self.compute_extras(coords, energy, instability)
        
        # Class bit: 1 for lesions
        class_bit = torch.ones(n_lesions, 1)
        
        return {
            'coords': coords,
            'velocity': velocity,
            'energy': energy,
            'instability': instability,
            'extras': extras,
            'class_bit': class_bit,
            'subclass_id': subclass_id
        }
    
    def combine_points(self, organs: Dict[str, torch.Tensor], 
                      lesions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Combine organ and lesion points into a single point cloud with shuffling.
        
        Args:
            organs: Organ properties dictionary
            lesions: Lesion properties dictionary
            
        Returns:
            Combined and shuffled point cloud properties
        """
        combined = {}
        for key in organs.keys():
            combined[key] = torch.cat([organs[key], lesions[key]], dim=0)
        
        # Shuffle all points together to mix organs and lesions
        n_total = combined[list(combined.keys())[0]].shape[0]
        shuffle_indices = torch.randperm(n_total)
        
        # Apply the same shuffle to all properties
        for key in combined.keys():
            combined[key] = combined[key][shuffle_indices]
            
        return combined
    
    def get_nearest_organ_effect(self, lesion_coord: torch.Tensor, organ_coords: torch.Tensor, 
                                organ_subclasses: torch.Tensor) -> Tuple[float, float]:
        """
        Get the effect of the nearest organ on a lesion.
        
        Args:
            lesion_coord: Lesion coordinates, shape [3]
            organ_coords: All organ coordinates, shape [N_organs, 3]
            organ_subclasses: All organ subclass IDs, shape [N_organs]
            
        Returns:
            Tuple of (gamma_multiplier, instability_multiplier)
        """
        if len(organ_coords) == 0:
            return 1.0, 1.0  # No effect if no organs
        
        # Calculate distances to all organs
        distances = torch.norm(organ_coords - lesion_coord, dim=1)
        
        # Find organs within their effective radius
        affected_by = []
        for i, (dist, subclass_id) in enumerate(zip(distances, organ_subclasses)):
            subclass_id = int(subclass_id.item())
            radius = self.organ_radii[subclass_id]
            if dist <= radius:
                affected_by.append((dist.item(), subclass_id))
        
        if not affected_by:
            return 1.0, 1.0  # No organs in range
        
        # Find the nearest organ (if ties, pick any)
        nearest_dist, nearest_subclass = min(affected_by, key=lambda x: x[0])
        
        # Apply the nearest organ's effect
        gamma_multiplier = 1.0
        instability_multiplier = 1.0
        
        if nearest_subclass in self.organ_gamma_reducers:
            gamma_multiplier = self.organ_gamma_effects[nearest_subclass]
        
        if nearest_subclass in self.organ_instability_boosters:
            instability_multiplier = self.organ_instability_effects[nearest_subclass]
        
        return gamma_multiplier, instability_multiplier
    
    def deterministic_step(self, points: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Execute one deterministic time step with subclass-dependent dynamics.
        
        Args:
            points: Current point cloud state
            
        Returns:
            Updated point cloud state after one time step
        """
        # Deep copy to avoid modifying input
        new_points = {}
        for key, value in points.items():
            new_points[key] = value.clone()
        
        n_points = new_points['coords'].shape[0]
        points_to_delete = []
        points_to_add = []
        
        # Separate organs and lesions for interaction calculations
        is_lesion = new_points['class_bit'].squeeze() == 1.0
        is_organ = ~is_lesion
        
        organ_coords = new_points['coords'][is_organ] if is_organ.any() else torch.empty(0, 3)
        organ_subclasses = new_points['subclass_id'][is_organ].squeeze() if is_organ.any() else torch.empty(0)
        
        lesion_coords = new_points['coords'][is_lesion] if is_lesion.any() else torch.empty(0, 3)
        lesion_subclasses = new_points['subclass_id'][is_lesion].squeeze() if is_lesion.any() else torch.empty(0)
        
        for i in range(n_points):
            # Position update for all points
            new_points['coords'][i] += new_points['velocity'][i] * self.DELTA_T
            
            # Add positional noise
            pos_noise = torch.normal(0, self.SIGMA_POS, size=(3,))
            new_points['coords'][i] += pos_noise
            
            # Lesion-specific rules (class_bit == 1)
            if new_points['class_bit'][i].item() == 1.0:
                subclass_id = int(new_points['subclass_id'][i].item())
                
                # Get organ effects on this lesion
                gamma_multiplier, instability_multiplier = self.get_nearest_organ_effect(
                    new_points['coords'][i], organ_coords, organ_subclasses
                )
                
                # Apply organ effect on instability
                modified_instability = new_points['instability'][i] * instability_multiplier
                new_points['instability'][i] = torch.clamp(modified_instability, 0, 1)
                
                # Energy update with subclass-specific parameters and organ effects
                nutrient = self.nutrient_field(
                    new_points['coords'][i], lesion_coords, lesion_subclasses, subclass_id
                )
                
                # Apply subclass-specific GAMMA_GAIN and organ effect
                base_gamma_gain = self.lesion_gamma_gain[subclass_id]
                effective_gamma_gain = base_gamma_gain * gamma_multiplier
                delta_decay = self.lesion_delta_decay[subclass_id]
                
                energy_change = effective_gamma_gain * nutrient - delta_decay
                new_energy = torch.clamp(new_points['energy'][i] + energy_change, 0, 1)
                new_points['energy'][i] = new_energy
                
                # Fate decision with subclass-specific thresholds
                death_threshold = self.lesion_energy_death_threshold[subclass_id]
                energy_dup_threshold = self.lesion_energy_dup_threshold[subclass_id]
                instability_dup_threshold = self.lesion_instability_dup_threshold[subclass_id]
                
                if new_energy < death_threshold:
                    # Mark for deletion
                    points_to_delete.append(i)
                elif (new_energy > energy_dup_threshold and 
                      new_points['instability'][i] > instability_dup_threshold and
                      subclass_id in self.initial_lesion_subclasses):  # Only initial lesions can duplicate
                      # ToDo: For now, lesions cannot duplicate twice bcz instability is not high enough
                      # but there is no rule to prevent it! 
                    
                    # Create child lesion
                    child_point = {}
                    for key in new_points.keys():
                        child_point[key] = new_points[key][i:i+1].clone()
                    
                    # Child position: parent + small noise
                    child_noise = torch.normal(0, 0.02, size=(1, 3))
                    child_point['coords'] += child_noise
                    
                    # Child velocity: parent + small noise
                    velocity_noise = torch.normal(0, 0.02, size=(1, 3))
                    child_point['velocity'] += velocity_noise
                    
                    # Split energy between parent and child (halved)
                    split_energy = 0.5 * new_energy
                    new_points['energy'][i] = split_energy
                    child_point['energy'][0] = split_energy
                    
                    # Split and reduce instability between parent and child
                    split_instability = 0.1 * new_points['instability'][i]
                    new_points['instability'][i] = split_instability
                    child_point['instability'][0] = split_instability
                    
                    # Child gets evolved subclass via transition map
                    if subclass_id in self.subclass_transition_map:
                        child_subclass = self.subclass_transition_map[subclass_id]
                        child_point['subclass_id'][0] = child_subclass
                    
                    # Recompute child's extras with new parameters
                    child_extras = self.compute_extras(
                        child_point['coords'], 
                        child_point['energy'], 
                        child_point['instability']
                    )
                    child_point['extras'] = child_extras
                    
                    points_to_add.append(child_point)
            
            # Recompute extras for this point (organs and lesions)
            coords = new_points['coords'][i:i+1]
            energy = new_points['energy'][i:i+1]
            instability = new_points['instability'][i:i+1]
            new_points['extras'][i] = self.compute_extras(coords, energy, instability).squeeze(0)
        
        # Remove points marked for deletion (in reverse order to maintain indices)
        for idx in sorted(points_to_delete, reverse=True):
            for key in new_points.keys():
                new_points[key] = torch.cat([
                    new_points[key][:idx],
                    new_points[key][idx+1:]
                ], dim=0)
        
        # Add duplicated points
        for child_point in points_to_add:
            for key in new_points.keys():
                new_points[key] = torch.cat([new_points[key], child_point[key]], dim=0)
        
        return new_points
    
    def create_feature_vector(self, points: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Create the 64-dimensional feature vector.
        
        Args:
            points: Point cloud properties
            
        Returns:
            Feature vectors, shape [N, 64]
        """
        # Feature vector components (coordinates are separate)
        velocity = points['velocity']  # [N, 3] - indices 0-2
        energy = points['energy']  # [N, 1] - index 3
        instability = points['instability']  # [N, 1] - index 4
        extras = points['extras']  # [N, 59] - indices 5-63
        
        # Concatenate all features (without coordinates and without class_bit)
        features = torch.cat([
            velocity,    # 0-2
            energy,      # 3
            instability, # 4
            extras,      # 5-63
        ], dim=-1)
        
        return features
    
    def generate_patient(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate a complete triple-timepoint patient with enhanced subclass dynamics.
        
        Returns:
            Tuple of (coords_list, feats_list, labels_list) for T0, T1, T2
        """
        # Step 1: Decide counts respecting subclass constraints
        max_initial_lesion = len(self.initial_lesion_subclasses)  # Available initial lesion subclasses
        
        n_organs = np.random.randint(self.MIN_ORGANS, self.MAX_ORGANS + 1)
        n_lesions = np.random.randint(self.MIN_LESIONS//2, max_initial_lesion + 1) # since lesions are divide into 'initial' and 'evolved'
        
        # Step 2 & 3: Initialize organs and lesions
        organs = self.initialize_organs(n_organs)
        lesions = self.initialize_lesions(n_lesions) # These are 'initial' lesions
        
        # Combine into initial point cloud
        points_t0 = self.combine_points(organs, lesions)
        
        # Step 4: Execute deterministic steps
        points_t1 = self.deterministic_step(points_t0)
        points_t2 = self.deterministic_step(points_t1)
        
        # Create output tensors for each timepoint
        coords_list = []
        feats_list = []
        labels_list = []
        
        for points in [points_t0, points_t1, points_t2]:
            coords = points['coords']
            features = self.create_feature_vector(points)
            # Combine class_bit and subclass_id into labels
            class_bits = points['class_bit'].squeeze(-1)  # [N]
            subclass_ids = points['subclass_id'].squeeze(-1)  # [N]
            
            coords_list.append(coords)
            feats_list.append(features)
            labels_list.append((class_bits, subclass_ids))
        
        return coords_list, feats_list, labels_list
    
    def generate_dataset(self, n_patients: int = 1000, output_dir: str = "synthetic_dataset"):
        """
        Generate the complete dataset with TP0, TP1, TP2 timepoint folders.
        
        Args:
            n_patients: Number of patients to generate
            output_dir: Output directory for dataset files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timepoint directories
        tp_dirs = {}
        for t in range(3):
            tp_dir = os.path.join(output_dir, f"TP{t}")
            os.makedirs(tp_dir, exist_ok=True)
            tp_dirs[t] = tp_dir
        
        print(f"Generating {n_patients} patients...")
        
        # Reset random seed for reproducibility
        self.set_random_seed(self.seed)
        
        for patient_id in range(n_patients):
            if (patient_id + 1) % 1000 == 0:
                print(f"  Generated {patient_id + 1}/{n_patients} patients")
            
            # Generate patient data
            coords_list, feats_list, labels_list = self.generate_patient()
            
            # Save three timepoint files (one in each TP folder)
            for t, (coords, feats, (class_bits, subclass_ids)) in enumerate(zip(coords_list, feats_list, labels_list)):
                # Concatenate coords, feats, class_bits, and subclass_ids into a single tensor
                # coords: [N, 3], feats: [N, 64], class_bits: [N], subclass_ids: [N] -> [N, 69]
                class_bits_expanded = class_bits.unsqueeze(-1)  # [N, 1]
                subclass_ids_expanded = subclass_ids.unsqueeze(-1)  # [N, 1]
                patient_tensor = torch.cat([coords, feats, class_bits_expanded, subclass_ids_expanded], dim=-1)  # [N, 69]
                
                filename = f"{patient_id}.pt"
                filepath = os.path.join(tp_dirs[t], filename)
                torch.save(patient_tensor, filepath)
        
        print(f"Completed dataset generation: {n_patients} patients")
        
        print(f"\nDataset generation complete! Saved to: {output_dir}")


def main():
    print("Synthetic Dataset Generator - Rule-Driven Points in a Box")
    print("=========================================================")
    
    # Create generator with fixed seed for reproducibility
    generator = SyntheticDatasetGenerator(seed=0)
    
    # Generate dataset
    print("Generating dataset...")
    generator.generate_dataset(
        n_patients=10000,
        output_dir="data/synthetic_dataset"
    )


if __name__ == "__main__":
    main()
