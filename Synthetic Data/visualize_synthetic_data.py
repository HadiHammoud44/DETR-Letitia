"""
Comprehensive Visualization Tool for Synthetic Dataset Generator

This tool provides interactive 3D visualization of point trajectories, dynamics,
and validation of the synthetic dataset generator's complex behaviors.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, CheckButtons
import torch
import sys
import os
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

sys.path.append('/mnt/letitia/scratch/students/hhammoud/detr')
from synthetic_dataset_generator import SyntheticDatasetGenerator

class SyntheticDataVisualizer:
    """
    Comprehensive visualization tool for synthetic dataset analysis.
    """
    
    def __init__(self, generator: SyntheticDatasetGenerator, seed: int = 42):
        """
        Initialize the visualizer.
        
        Args:
            generator: Configured SyntheticDatasetGenerator instance
            seed: Random seed for reproducible visualizations
        """
        self.generator = generator
        self.seed = seed
        
        # Generate sample data for visualization
        self.coords_list, self.feats_list, self.labels_list = self._generate_sample_data()
        
        # Set up color schemes
        self._setup_color_schemes()
        
        # Current visualization state
        self.current_timepoint = 0
        self.show_organs = True
        self.show_lesions = True
        self.show_trajectories = True
        self.show_field_centers = True
        self.show_organ_effects = True
        
    def _generate_sample_data(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Tuple]]:
        """Generate sample patient data for visualization."""
        self.generator.set_random_seed(self.seed)
        return self.generator.generate_patient()
    
    def _setup_color_schemes(self):
        """Set up color schemes for different elements."""
        # Organ colors (consistent across time)
        self.organ_colors = plt.cm.Set1(np.linspace(0, 1, len(self.generator.organ_radii)))
        
        # Lesion colors by subclass - evolved lesions inherit parent colors
        initial_subclasses = list(self.generator.initial_lesion_subclasses)
        evolved_subclasses = list(self.generator.evolved_lesion_subclasses)
        
        # Assign colors to initial subclasses first
        initial_colors = plt.cm.Set2(np.linspace(0, 1, len(initial_subclasses)))
        self.lesion_color_map = {subclass: color for subclass, color in zip(initial_subclasses, initial_colors)}
        
        # Evolved lesions inherit their parent's color via transition map
        for initial_id, evolved_id in self.generator.subclass_transition_map.items():
            if initial_id in self.lesion_color_map:
                self.lesion_color_map[evolved_id] = self.lesion_color_map[initial_id]
        
        # Handle any evolved lesions without parents (fallback colors)
        orphaned_evolved = [sc for sc in evolved_subclasses if sc not in self.lesion_color_map]
        if orphaned_evolved:
            fallback_colors = plt.cm.Set3(np.linspace(0, 1, len(orphaned_evolved)))
            for subclass, color in zip(orphaned_evolved, fallback_colors):
                self.lesion_color_map[subclass] = color
        
        # Special colors for field centers
        self.c1_color = 'red'
        self.c2_color = 'blue'
        
        # Event colors
        self.death_color = 'black'
        self.birth_color = 'gold'
        self.trajectory_color = 'gray'
    
    def plot_3d_trajectories(self, figsize=(15, 10), save_path=None):
        """
        Create comprehensive 3D trajectory visualization.
        
        Args:
            figsize: Figure size tuple
            save_path: Optional path to save the figure
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot organ trajectories (should be persistent)
        self._plot_organ_trajectories(ax)
        
        # Plot lesion trajectories with subclass colors
        self._plot_lesion_trajectories(ax)
        
        # Plot field centers
        if self.show_field_centers:
            self._plot_field_centers(ax)
        
        # Plot organ effect zones
        if self.show_organ_effects:
            self._plot_organ_effect_zones(ax)
        
        # Customize plot
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.set_title('3D Point Trajectories (T0 → T1 → T2)')
        
        # Add legend
        self._add_3d_legend(ax)
        
        # Set equal aspect ratio
        self._set_equal_aspect_3d(ax)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close(fig)  # Close figure to free memory
    
    def _plot_organ_trajectories(self, ax):
        """Plot organ trajectories in 3D."""
        for t in range(3):
            class_bits, subclass_ids = self.labels_list[t]
            coords = self.coords_list[t]
            
            # Find organs
            organ_mask = class_bits == 0
            if organ_mask.any():
                organ_coords = coords[organ_mask]
                organ_subclass_ids = subclass_ids[organ_mask]
                
                for i, (coord, subclass) in enumerate(zip(organ_coords, organ_subclass_ids)):
                    subclass_int = int(subclass.item())
                    color = self.organ_colors[subclass_int % len(self.organ_colors)]
                    
                    # Plot point
                    ax.scatter(coord[0], coord[1], coord[2], 
                             c=[color], s=100, marker='s', alpha=0.8,
                             edgecolors='black', linewidth=1)
                    
                    # Add trajectory lines between timepoints
                    if t > 0 and self.show_trajectories:
                        # Find same organ in previous timepoint
                        prev_coords = self._find_organ_in_previous_timepoint(subclass_int, t-1)
                        if prev_coords is not None:
                            ax.plot([prev_coords[0], coord[0]], 
                                   [prev_coords[1], coord[1]], 
                                   [prev_coords[2], coord[2]], 
                                   color=self.trajectory_color, alpha=0.6, linewidth=1)
    
    def _plot_lesion_trajectories(self, ax):
        """Plot lesion trajectories with subclass-specific colors."""
        # Track lesions across timepoints for trajectory plotting
        lesion_trajectories = {}
        
        for t in range(3):
            class_bits, subclass_ids = self.labels_list[t]
            coords = self.coords_list[t]
            
            # Find lesions
            lesion_mask = class_bits == 1
            if lesion_mask.any():
                lesion_coords = coords[lesion_mask]
                lesion_subclass_ids = subclass_ids[lesion_mask]
                
                for coord, subclass in zip(lesion_coords, lesion_subclass_ids):
                    subclass_int = int(subclass.item())
                    color = self.lesion_color_map.get(subclass_int, 'purple')
                    
                    # Determine marker based on subclass type
                    if subclass_int in self.generator.initial_lesion_subclasses:
                        marker = 'o'  # Circle for initial
                    else:
                        marker = '^'  # Triangle for evolved
                    
                    # Plot point
                    ax.scatter(coord[0], coord[1], coord[2], 
                             c=[color], s=80, marker=marker, alpha=0.7,
                             edgecolors='black', linewidth=0.5)
                    
                    # Store for trajectory tracking
                    if subclass_int not in lesion_trajectories:
                        lesion_trajectories[subclass_int] = []
                    lesion_trajectories[subclass_int].append((t, coord))
        
        # Draw trajectory lines for lesions
        if self.show_trajectories:
            for subclass_int, trajectory in lesion_trajectories.items():
                if len(trajectory) > 1:
                    trajectory.sort(key=lambda x: x[0])  # Sort by time
                    coords_seq = [pos for _, pos in trajectory]
                    
                    for i in range(len(coords_seq) - 1):
                        curr_coord = coords_seq[i]
                        next_coord = coords_seq[i + 1]
                        ax.plot([curr_coord[0], next_coord[0]], 
                               [curr_coord[1], next_coord[1]], 
                               [curr_coord[2], next_coord[2]], 
                               color=self.trajectory_color, alpha=0.4, linewidth=1)
    
    def _plot_field_centers(self, ax):
        """Plot dynamic nutrient field centers C1 and C2."""
        for t in range(3):
            class_bits, subclass_ids = self.labels_list[t]
            coords = self.coords_list[t]
            
            # Find lesions
            lesion_mask = class_bits == 1
            if lesion_mask.any():
                lesion_coords = coords[lesion_mask]
                lesion_subclass_ids = subclass_ids[lesion_mask]
                
                # Calculate C1 and C2 centers
                c1_coords = []
                c2_coords = []
                
                for coord, subclass in zip(lesion_coords, lesion_subclass_ids):
                    subclass_int = int(subclass.item())
                    field_assignment = self.generator.lesion_field_assignment.get(subclass_int, 'C1')
                    
                    if field_assignment == 'C1':
                        c1_coords.append(coord)
                    else:
                        c2_coords.append(coord)
                
                # Plot centers
                if c1_coords:
                    c1_center = torch.stack(c1_coords).mean(dim=0)
                    ax.scatter(c1_center[0], c1_center[1], c1_center[2], 
                             c=self.c1_color, s=200, marker='*', alpha=0.8,
                             edgecolors='black', linewidth=2, label=f'C1 Center T{t}')
                
                if c2_coords:
                    c2_center = torch.stack(c2_coords).mean(dim=0)
                    ax.scatter(c2_center[0], c2_center[1], c2_center[2], 
                             c=self.c2_color, s=200, marker='*', alpha=0.8,
                             edgecolors='black', linewidth=2, label=f'C2 Center T{t}')
    
    def _plot_organ_effect_zones(self, ax):
        """Plot organ effect zones as wireframe spheres."""
        for t in range(3):
            class_bits, subclass_ids = self.labels_list[t]
            coords = self.coords_list[t]
            
            # Find organs
            organ_mask = class_bits == 0
            if organ_mask.any():
                organ_coords = coords[organ_mask]
                organ_subclass_ids = subclass_ids[organ_mask]
                
                for coord, subclass in zip(organ_coords, organ_subclass_ids):
                    subclass_int = int(subclass.item())
                    radius = self.generator.organ_radii[subclass_int]
                    color = self.organ_colors[subclass_int % len(self.organ_colors)]
                    
                    # Create wireframe sphere
                    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                    x = radius * np.cos(u) * np.sin(v) + coord[0].item()
                    y = radius * np.sin(u) * np.sin(v) + coord[1].item()
                    z = radius * np.cos(v) + coord[2].item()
                    
                    ax.plot_wireframe(x, y, z, alpha=0.05, color=color)
    
    def _find_organ_in_previous_timepoint(self, subclass_id: int, timepoint: int) -> Optional[torch.Tensor]:
        """Find organ with given subclass in previous timepoint."""
        if timepoint < 0:
            return None
        
        class_bits, subclass_ids = self.labels_list[timepoint]
        coords = self.coords_list[timepoint]
        
        organ_mask = class_bits == 0
        if organ_mask.any():
            organ_coords = coords[organ_mask]
            organ_subclass_ids = subclass_ids[organ_mask]
            
            for coord, subclass in zip(organ_coords, organ_subclass_ids):
                if int(subclass.item()) == subclass_id:
                    return coord
        return None
    
    def _add_3d_legend(self, ax):
        """Add legend to 3D plot."""
        legend_elements = []
        
        # Organ legend
        legend_elements.append(mpatches.Patch(color='gray', label='Organs (squares)'))
        
        # Lesion legend
        legend_elements.append(mpatches.Patch(color='lightblue', label='Initial Lesions (circles)'))
        legend_elements.append(mpatches.Patch(color='lightcoral', label='Evolved Lesions (triangles)'))
        
        # Field centers
        legend_elements.append(mpatches.Patch(color=self.c1_color, label='C1 Field Center'))
        legend_elements.append(mpatches.Patch(color=self.c2_color, label='C2 Field Center'))
        
        ax.legend(handles=legend_elements, loc='upper right')
    
    def _set_equal_aspect_3d(self, ax):
        """Set equal aspect ratio for 3D plot."""
        # Get the range of each axis
        all_coords = torch.cat(self.coords_list, dim=0)
        max_range = torch.max(torch.abs(all_coords)) * 1.1
        
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
    
    def plot_energy_dynamics(self, figsize=(15, 8), save_path=None):
        """
        Plot energy and instability dynamics over time.
        
        Args:
            figsize: Figure size tuple
            save_path: Optional path to save the figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Track lesion energies and instabilities
        lesion_energies = {}
        lesion_instabilities = {}
        
        for t in range(3):
            class_bits, subclass_ids = self.labels_list[t]
            features = self.feats_list[t]
            
            lesion_mask = class_bits == 1
            if lesion_mask.any():
                lesion_features = features[lesion_mask]
                lesion_subclass_ids = subclass_ids[lesion_mask]
                
                energies = lesion_features[:, 3]  # Energy is at index 3
                instabilities = lesion_features[:, 4]  # Instability is at index 4
                
                for energy, instability, subclass in zip(energies, instabilities, lesion_subclass_ids):
                    subclass_int = int(subclass.item())
                    
                    if subclass_int not in lesion_energies:
                        lesion_energies[subclass_int] = []
                        lesion_instabilities[subclass_int] = []
                    
                    lesion_energies[subclass_int].append((t, energy.item()))
                    lesion_instabilities[subclass_int].append((t, instability.item()))
        
        # Plot energy trajectories
        for subclass_int, energy_trajectory in lesion_energies.items():
            times = [t for t, _ in energy_trajectory]
            energies = [e for _, e in energy_trajectory]
            
            color = self.lesion_color_map.get(subclass_int, 'purple')
            ax1.plot(times, energies, 'o-', color=color, alpha=0.7, 
                    label=f'Subclass {subclass_int}')
        
        ax1.set_xlabel('Timepoint')
        ax1.set_ylabel('Energy')
        ax1.set_title('Lesion Energy Dynamics')
        ax1.grid(True, alpha=0.3)
        # ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot instability trajectories
        for subclass_int, instability_trajectory in lesion_instabilities.items():
            times = [t for t, _ in instability_trajectory]
            instabilities = [i for _, i in instability_trajectory]
            
            color = self.lesion_color_map.get(subclass_int, 'purple')
            ax2.plot(times, instabilities, 'o-', color=color, alpha=0.7,
                    label=f'Subclass {subclass_int}')
        
        ax2.set_xlabel('Timepoint')
        ax2.set_ylabel('Instability')
        ax2.set_title('Lesion Instability Dynamics')
        ax2.grid(True, alpha=0.3)
        # ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot energy distribution by timepoint
        for t in range(3):
            class_bits, _ = self.labels_list[t]
            features = self.feats_list[t]
            
            lesion_mask = class_bits == 1
            if lesion_mask.any():
                energies = features[lesion_mask, 3]
                ax3.hist(energies.numpy(), bins=10, alpha=0.6, label=f'T{t}')
        
        ax3.set_xlabel('Energy')
        ax3.set_ylabel('Count')
        ax3.set_title('Energy Distribution by Timepoint')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot thresholds and parameters
        subclasses = list(lesion_energies.keys())
        death_thresholds = [self.generator.lesion_energy_death_threshold[sc] for sc in subclasses]
        dup_thresholds = [self.generator.lesion_energy_dup_threshold[sc] for sc in subclasses]
        
        x_pos = np.arange(len(subclasses))
        width = 0.35
        
        ax4.bar(x_pos - width/2, death_thresholds, width, label='Death Threshold', alpha=0.7)
        ax4.bar(x_pos + width/2, dup_thresholds, width, label='Duplication Threshold', alpha=0.7)
        
        ax4.set_xlabel('Subclass ID')
        ax4.set_ylabel('Threshold')
        ax4.set_title('Energy Thresholds by Subclass')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(subclasses)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close(fig)  # Close figure to free memory
    
    def plot_subclass_analysis(self, figsize=(16, 10), save_path=None):
        """
        Comprehensive subclass analysis dashboard.
        
        Args:
            figsize: Figure size tuple
            save_path: Optional path to save the figure
        """
        fig = plt.figure(figsize=figsize)
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Subclass transition visualization
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_subclass_transitions(ax1)
        
        # Parameter distributions
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_parameter_distributions(ax2)
        
        # Point counts over time
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_point_counts_over_time(ax3)
        
        # Field assignments
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_field_assignments(ax4)
        
        # Organ effects
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_organ_effects(ax5)
        
        # Event summary
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_event_summary(ax6)
        
        plt.suptitle('Comprehensive Subclass Analysis Dashboard', fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close(fig)  # Close figure to free memory
    
    def _plot_subclass_transitions(self, ax):
        """Plot subclass transition network."""
        # Create transition graph
        initial_subclasses = list(self.generator.initial_lesion_subclasses)
        evolved_subclasses = list(self.generator.evolved_lesion_subclasses)
        
        # Position initial subclasses on left, evolved on right
        n_initial = len(initial_subclasses)
        n_evolved = len(evolved_subclasses)
        
        initial_y = np.linspace(0, 1, n_initial)
        evolved_y = np.linspace(0, 1, n_evolved)
        
        # Plot initial subclasses
        for i, subclass in enumerate(initial_subclasses):
            ax.scatter(0, initial_y[i], s=100, c='lightblue', marker='o', 
                      edgecolors='black', linewidth=1)
            ax.text(-0.1, initial_y[i], str(subclass), ha='right', va='center')
        
        # Plot evolved subclasses
        for i, subclass in enumerate(evolved_subclasses):
            ax.scatter(1, evolved_y[i], s=100, c='lightcoral', marker='^', 
                      edgecolors='black', linewidth=1)
            ax.text(1.1, evolved_y[i], str(subclass), ha='left', va='center')
        
        # Draw transition arrows
        for initial_id, evolved_id in self.generator.subclass_transition_map.items():
            if initial_id in initial_subclasses and evolved_id in evolved_subclasses:
                i_idx = initial_subclasses.index(initial_id)
                e_idx = evolved_subclasses.index(evolved_id)
                
                ax.arrow(0.05, initial_y[i_idx], 0.9, evolved_y[e_idx] - initial_y[i_idx],
                        head_width=0.02, head_length=0.05, fc='gray', ec='gray', alpha=0.7)
        
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title('Subclass Transition Map')
        ax.set_xlabel('Initial → Evolved')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Initial', 'Evolved'])
        ax.grid(True, alpha=0.3)
    
    def _plot_parameter_distributions(self, ax):
        """Plot parameter distributions for subclasses."""
        # Collect parameters
        subclasses = list(self.generator.initial_lesion_subclasses) + list(self.generator.evolved_lesion_subclasses)
        gamma_gains = [self.generator.lesion_gamma_gain[sc] for sc in subclasses]
        delta_decays = [self.generator.lesion_delta_decay[sc] for sc in subclasses]
        
        # Create scatter plot
        ax.scatter(gamma_gains, delta_decays, c=range(len(subclasses)), 
                  cmap='viridis', s=80, alpha=0.7, edgecolors='black')
        
        # Add subclass labels
        for i, sc in enumerate(subclasses):
            ax.annotate(str(sc), (gamma_gains[i], delta_decays[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('GAMMA_GAIN')
        ax.set_ylabel('DELTA_DECAY')
        ax.set_title('Subclass Parameter Distribution')
        ax.grid(True, alpha=0.3)
    
    def _plot_point_counts_over_time(self, ax):
        """Plot point counts over time."""
        timepoints = [0, 1, 2]
        organ_counts = []
        lesion_counts = []
        total_counts = []
        
        for t in range(3):
            class_bits, _ = self.labels_list[t]
            
            organ_count = (class_bits == 0).sum().item()
            lesion_count = (class_bits == 1).sum().item()
            total_count = len(class_bits)
            
            organ_counts.append(organ_count)
            lesion_counts.append(lesion_count)
            total_counts.append(total_count)
        
        width = 0.25
        x = np.array(timepoints)
        
        ax.bar(x - width, organ_counts, width, label='Organs', alpha=0.8, color='lightblue')
        ax.bar(x, lesion_counts, width, label='Lesions', alpha=0.8, color='lightcoral')
        ax.bar(x + width, total_counts, width, label='Total', alpha=0.8, color='lightgray')
        
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('Count')
        ax.set_title('Point Counts Over Time')
        ax.set_xticks(timepoints)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_field_assignments(self, ax):
        """Plot field assignments (C1/C2)."""
        c1_subclasses = list(self.generator.c1_lesion_subclasses)
        c2_subclasses = list(self.generator.c2_lesion_subclasses)
        
        # Count assignments
        c1_count = len(c1_subclasses)
        c2_count = len(c2_subclasses)
        
        # Pie chart
        sizes = [c1_count, c2_count]
        colors = [self.c1_color, self.c2_color]
        labels = [f'C1 Field ({c1_count})', f'C2 Field ({c2_count})']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        ax.set_title('Lesion Field Assignments')
    
    def _plot_organ_effects(self, ax):
        """Plot organ effect types and ranges."""
        instability_boosters = list(self.generator.organ_instability_boosters)
        gamma_reducers = list(self.generator.organ_gamma_reducers)
        
        # Get radii for each type
        booster_radii = [self.generator.organ_radii[sc] for sc in instability_boosters]
        reducer_radii = [self.generator.organ_radii[sc] for sc in gamma_reducers]
        
        # Box plot
        data = [booster_radii, reducer_radii]
        labels = ['Instability\nBoosters', 'Gamma\nReducers']
        
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('orange')
        bp['boxes'][1].set_facecolor('lightgreen')
        
        ax.set_ylabel('Effect Radius')
        ax.set_title('Organ Effect Ranges by Type')
        ax.grid(True, alpha=0.3)
    
    def _plot_event_summary(self, ax):
        """Plot summary of events (deaths, duplications)."""
        # Count events by comparing timepoints
        death_events = 0
        duplication_events = 0
        
        # T0 to T1
        t0_lesions = set()
        class_bits_t0, subclass_ids_t0 = self.labels_list[0]
        lesion_mask_t0 = class_bits_t0 == 1
        if lesion_mask_t0.any():
            for subclass in subclass_ids_t0[lesion_mask_t0]:
                t0_lesions.add(int(subclass.item()))
        
        t1_lesions = set()
        class_bits_t1, subclass_ids_t1 = self.labels_list[1]
        lesion_mask_t1 = class_bits_t1 == 1
        if lesion_mask_t1.any():
            for subclass in subclass_ids_t1[lesion_mask_t1]:
                t1_lesions.add(int(subclass.item()))
        
        # Count deaths (disappeared) and births (new evolved subclasses)
        deaths_t0_t1 = len(t0_lesions - t1_lesions)
        births_t0_t1 = len(t1_lesions - t0_lesions)
        
        # Similar for T1 to T2
        t2_lesions = set()
        class_bits_t2, subclass_ids_t2 = self.labels_list[2]
        lesion_mask_t2 = class_bits_t2 == 1
        if lesion_mask_t2.any():
            for subclass in subclass_ids_t2[lesion_mask_t2]:
                t2_lesions.add(int(subclass.item()))
        
        deaths_t1_t2 = len(t1_lesions - t2_lesions)
        births_t1_t2 = len(t2_lesions - t1_lesions)
        
        # Plot events
        transitions = ['T0→T1', 'T1→T2']
        deaths = [deaths_t0_t1, deaths_t1_t2]
        births = [births_t0_t1, births_t1_t2]
        
        x = np.arange(len(transitions))
        width = 0.35
        
        ax.bar(x - width/2, deaths, width, label='Deaths', color=self.death_color, alpha=0.7)
        ax.bar(x + width/2, births, width, label='Births', color=self.birth_color, alpha=0.7)
        
        ax.set_xlabel('Transition')
        ax.set_ylabel('Count')
        ax.set_title('Lesion Events')
        ax.set_xticks(x)
        ax.set_xticklabels(transitions)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def create_interactive_animation(self, figsize=(12, 8), interval=1000, save_path=None):
        """
        Create animated visualization of temporal evolution.
        
        Args:
            figsize: Figure size tuple
            interval: Animation interval in milliseconds
            save_path: Optional path to save animation as GIF
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        def animate(frame):
            ax.clear()
            
            # Current timepoint
            t = frame % 3
            class_bits, subclass_ids = self.labels_list[t]
            coords = self.coords_list[t]
            
            # Plot organs
            organ_mask = class_bits == 0
            if organ_mask.any():
                organ_coords = coords[organ_mask]
                organ_subclass_ids = subclass_ids[organ_mask]
                
                for coord, subclass in zip(organ_coords, organ_subclass_ids):
                    subclass_int = int(subclass.item())
                    color = self.organ_colors[subclass_int % len(self.organ_colors)]
                    ax.scatter(coord[0], coord[1], coord[2], 
                             c=[color], s=100, marker='s', alpha=0.8)
            
            # Plot lesions
            lesion_mask = class_bits == 1
            if lesion_mask.any():
                lesion_coords = coords[lesion_mask]
                lesion_subclass_ids = subclass_ids[lesion_mask]
                
                for coord, subclass in zip(lesion_coords, lesion_subclass_ids):
                    subclass_int = int(subclass.item())
                    color = self.lesion_color_map.get(subclass_int, 'purple')
                    
                    if subclass_int in self.generator.initial_lesion_subclasses:
                        marker = 'o'
                    else:
                        marker = '^'
                    
                    ax.scatter(coord[0], coord[1], coord[2], 
                             c=[color], s=80, marker=marker, alpha=0.7)
            
            # Set labels and title
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_zlabel('Z Coordinate')
            ax.set_title(f'Point Cloud at Timepoint T{t}')
            
            # Set consistent limits
            self._set_equal_aspect_3d(ax)
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=9, interval=interval, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=1)
        
        plt.close(fig)  # Close figure to free memory
        return anim
    
    def generate_validation_report(self, save_path=None):
        """
        Generate comprehensive validation report.
        
        Args:
            save_path: Optional path to save the report figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. Energy conservation during duplication
        self._validate_energy_conservation(axes[0])
        
        # 2. Organ persistence
        self._validate_organ_persistence(axes[1])
        
        # 3. Subclass uniqueness
        self._validate_subclass_uniqueness(axes[2])
        
        # 4. Velocity distribution
        self._validate_velocity_distribution(axes[3])
        
        # 5. Field center dynamics
        self._validate_field_centers(axes[4])
        
        # 6. Parameter consistency
        self._validate_parameter_consistency(axes[5])
        
        plt.suptitle('Synthetic Dataset Validation Report', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close(fig)  # Close figure to free memory
    
    def _validate_energy_conservation(self, ax):
        """Validate energy conservation during duplication."""
        # This is a conceptual validation - in practice we'd track specific duplication events
        energy_changes = []
        
        for t in range(2):
            class_bits_t, _ = self.labels_list[t]
            class_bits_t1, _ = self.labels_list[t+1]
            
            features_t = self.feats_list[t]
            features_t1 = self.feats_list[t+1]
            
            lesion_mask_t = class_bits_t == 1
            lesion_mask_t1 = class_bits_t1 == 1
            
            if lesion_mask_t.any() and lesion_mask_t1.any():
                energies_t = features_t[lesion_mask_t, 3]
                energies_t1 = features_t1[lesion_mask_t1, 3]
                
                total_energy_t = energies_t.sum().item()
                total_energy_t1 = energies_t1.sum().item()
                
                energy_changes.append(total_energy_t1 - total_energy_t)
        
        ax.bar(['T0→T1', 'T1→T2'], energy_changes, alpha=0.7)
        ax.set_ylabel('Total Energy Change')
        ax.set_title('Energy Changes Between Timepoints')
        ax.grid(True, alpha=0.3)
    
    def _validate_organ_persistence(self, ax):
        """Validate that organs persist across timepoints."""
        organ_counts = []
        
        for t in range(3):
            class_bits, _ = self.labels_list[t]
            organ_count = (class_bits == 0).sum().item()
            organ_counts.append(organ_count)
        
        ax.plot([0, 1, 2], organ_counts, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('Organ Count')
        ax.set_title('Organ Persistence Validation')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(organ_counts) * 1.1)
    
    def _validate_subclass_uniqueness(self, ax):
        """Validate subclass uniqueness within timepoints."""
        uniqueness_scores = []
        
        for t in range(3):
            _, subclass_ids = self.labels_list[t]
            total_points = len(subclass_ids)
            unique_subclasses = len(torch.unique(subclass_ids))
            uniqueness_score = unique_subclasses / total_points
            uniqueness_scores.append(uniqueness_score)
        
        ax.bar([0, 1, 2], uniqueness_scores, alpha=0.7, color='lightgreen')
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('Uniqueness Score')
        ax.set_title('Subclass Uniqueness (1.0 = all unique)')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
    
    def _validate_velocity_distribution(self, ax):
        """Validate velocity distribution."""
        all_velocities = []
        
        for t in range(3):
            features = self.feats_list[t]
            velocities = features[:, :3]  # First 3 components are velocity
            velocity_magnitudes = torch.norm(velocities, dim=1)
            all_velocities.extend(velocity_magnitudes.numpy())
        
        ax.hist(all_velocities, bins=20, alpha=0.7, color='skyblue')
        ax.set_xlabel('Velocity Magnitude')
        ax.set_ylabel('Frequency')
        ax.set_title('Velocity Distribution Validation')
        ax.grid(True, alpha=0.3)
    
    def _validate_field_centers(self, ax):
        """Validate field center calculations."""
        c1_distances = []
        c2_distances = []
        
        for t in range(3):
            class_bits, subclass_ids = self.labels_list[t]
            coords = self.coords_list[t]
            
            lesion_mask = class_bits == 1
            if lesion_mask.any():
                lesion_coords = coords[lesion_mask]
                lesion_subclass_ids = subclass_ids[lesion_mask]
                
                c1_coords = []
                c2_coords = []
                
                for coord, subclass in zip(lesion_coords, lesion_subclass_ids):
                    subclass_int = int(subclass.item())
                    field_assignment = self.generator.lesion_field_assignment.get(subclass_int, 'C1')
                    
                    if field_assignment == 'C1':
                        c1_coords.append(coord)
                    else:
                        c2_coords.append(coord)
                
                if c1_coords and c2_coords:
                    c1_center = torch.stack(c1_coords).mean(dim=0)
                    c2_center = torch.stack(c2_coords).mean(dim=0)
                    distance = torch.norm(c1_center - c2_center).item()
                    
                    c1_distances.append(torch.norm(c1_center).item())
                    c2_distances.append(torch.norm(c2_center).item())
        
        timepoints = list(range(len(c1_distances)))
        ax.plot(timepoints, c1_distances, 'o-', label='C1 Center Distance from Origin', color=self.c1_color)
        ax.plot(timepoints, c2_distances, 's-', label='C2 Center Distance from Origin', color=self.c2_color)
        
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('Distance from Origin')
        ax.set_title('Field Center Movement')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _validate_parameter_consistency(self, ax):
        """Validate parameter consistency across subclasses."""
        # Check that parameters are within expected ranges
        all_gamma_gains = list(self.generator.lesion_gamma_gain.values())
        all_delta_decays = list(self.generator.lesion_delta_decay.values())
        all_organ_radii = list(self.generator.organ_radii.values())
        
        # Box plot of parameter ranges
        data = [all_gamma_gains, all_delta_decays, all_organ_radii]
        labels = ['GAMMA_GAIN\n[0.3, 0.7]', 'DELTA_DECAY\n[0.05, 0.15]', 'Organ Radii\n[0.25, 0.5]']
        
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Parameter Value')
        ax.set_title('Parameter Range Validation')
        ax.grid(True, alpha=0.3)


def main():
    """Example usage of the visualization tool."""
    print("Synthetic Data Visualization Tool")
    print("=================================")
    
    # Create generator
    generator = SyntheticDatasetGenerator(seed=30)
    
    # Create visualizer
    visualizer = SyntheticDataVisualizer(generator, seed=30)
    
    print("Generating visualizations...")
    
    # Create output directory for visualization images
    import os
    output_dir = "visualization_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 3D Trajectory Plot
    print("1. Creating 3D trajectory visualization...")
    visualizer.plot_3d_trajectories(save_path=f"{output_dir}/3d_trajectories.png")
    
    # 2. Energy Dynamics
    print("2. Creating energy dynamics plot...")
    visualizer.plot_energy_dynamics(save_path=f"{output_dir}/energy_dynamics.png")
    
    # 3. Subclass Analysis Dashboard
    print("3. Creating subclass analysis dashboard...")
    visualizer.plot_subclass_analysis(save_path=f"{output_dir}/subclass_analysis.png")
    
    # 4. Validation Report
    print("4. Creating validation report...")
    visualizer.generate_validation_report(save_path=f"{output_dir}/validation_report.png")
    
    # 5. Interactive Animation (save as GIF)
    print("5. Creating animation...")
    try:
        anim = visualizer.create_interactive_animation(save_path=f"{output_dir}/animation.gif")
        plt.close('all')  # Close animation figure
    except Exception as e:
        print(f"   Animation creation failed (expected in headless mode): {e}")
    
    print(f"Visualization complete! All plots saved to {output_dir}/ directory")
    print("\nGenerated files:")
    for file in os.listdir(f"{output_dir}"):
        print(f"  - {output_dir}/{file}")


if __name__ == "__main__":
    main()
