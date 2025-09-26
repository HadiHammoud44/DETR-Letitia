# Synthetic Dataset Visualization Tool - Usage Guide

This guide explains how to use the comprehensive visualization tool for validating the enhanced synthetic dataset generator.

## Quick Start

```python
from synthetic_dataset_generator import SyntheticDatasetGenerator
from visualize_synthetic_data import SyntheticDataVisualizer

# Create generator and visualizer
generator = SyntheticDatasetGenerator(seed=42)
visualizer = SyntheticDataVisualizer(generator, seed=42)

# Generate all visualizations
visualizer.plot_3d_trajectories(save_path="trajectories.png")
visualizer.plot_energy_dynamics(save_path="energy.png")
visualizer.plot_subclass_analysis(save_path="subclass.png")
visualizer.generate_validation_report(save_path="validation.png")
visualizer.create_interactive_animation(save_path="animation.gif")
```

## Visualization Features

### 1. 3D Trajectory Visualization (`plot_3d_trajectories`)

**Purpose**: Visualize point movements across T0 → T1 → T2 with trajectory lines.

**Key Elements**:
- **Organs**: Square markers with unique colors per subclass
- **Initial Lesions**: Circle markers with subclass-specific colors  
- **Evolved Lesions**: Triangle markers with inherited parent colors (same color as their initial lesion parent)
- **Field Centers**: Star markers (red for C1, blue for C2)
- **Organ Effect Zones**: Optional wireframe spheres showing radius effects
- **Trajectory Lines**: Gray lines connecting same points across time

**Validation Purpose**: 
- Verify organ persistence (organs should never disappear)
- Observe lesion trajectories and evolution events
- Check field center movements based on lesion positions
- Validate organ effect zones are reasonable

### 2. Energy Dynamics Analysis (`plot_energy_dynamics`)

**Purpose**: Track energy and instability changes over time for validation.

**Four Subplots**:
1. **Energy Trajectories**: Individual lesion energy over time by subclass
2. **Instability Trajectories**: Individual lesion instability over time by subclass  
3. **Energy Distribution**: Histogram of energy values at each timepoint
4. **Threshold Visualization**: Death vs duplication thresholds by subclass

**Validation Purpose**:
- Ensure energy changes follow expected patterns
- Verify energy conservation during duplication events
- Check that death/duplication thresholds are being respected
- Validate subclass-specific parameter differences

### 3. Subclass Analysis Dashboard (`plot_subclass_analysis`)

**Purpose**: Comprehensive analysis of subclass configurations and behaviors.

**Six Panels**:
1. **Transition Map**: Initial → Evolved lesion subclass mappings
2. **Parameter Distribution**: GAMMA_GAIN vs DELTA_DECAY scatter plot
3. **Point Counts**: Organ/lesion counts over time
4. **Field Assignments**: C1 vs C2 lesion distribution (pie chart)
5. **Organ Effects**: Effect radius distributions by organ type
6. **Event Summary**: Death and birth events between timepoints

**Validation Purpose**:
- Verify subclass transition logic is working
- Check parameter distributions are within expected ranges
- Ensure field assignments are balanced
- Validate event frequencies are reasonable

### 4. Validation Report (`generate_validation_report`)

**Purpose**: Systematic validation of dataset generator correctness.

**Six Validation Checks**:
1. **Energy Conservation**: Total energy changes during duplication
2. **Organ Persistence**: Organ counts should remain constant
3. **Subclass Uniqueness**: Uniqueness score within timepoints
4. **Velocity Distribution**: Velocity magnitude histogram
5. **Field Center Movement**: C1/C2 center distances from origin
6. **Parameter Consistency**: Box plots of parameter ranges

**Validation Purpose**:
- Automated correctness checking
- Identify potential bugs or inconsistencies
- Verify physical constraints are maintained
- Ensure parameter values are within expected ranges (GAMMA_GAIN: 8.0-20.0, DELTA_DECAY: 0.1-0.35, Organ Radii: 0.3-0.7)

### 5. Interactive Animation (`create_interactive_animation`)

**Purpose**: Animated visualization of temporal evolution.

**Features**:
- 3D point cloud evolution over T0 → T1 → T2
- Repeating animation with consistent viewpoint
- Color-coded organs and lesions by subclass
- Saves as GIF for easy sharing

**Validation Purpose**:
- Intuitive understanding of temporal dynamics
- Easy identification of unusual behaviors
- Visual verification of point cloud evolution

## Advanced Usage Examples

### Multi-Patient Analysis

```python
# Compare multiple patients
seeds = [100, 200, 300]
for i, seed in enumerate(seeds):
    generator = SyntheticDatasetGenerator(seed=seed)
    visualizer = SyntheticDataVisualizer(generator, seed=seed)
    visualizer.plot_3d_trajectories(save_path=f"patient_{i+1}.png")
```

### Custom Visualization Settings

```python
# Enable organ effect zones
visualizer.show_organ_effects = True
visualizer.plot_3d_trajectories(save_path="with_effects.png")

# Disable trajectory lines
visualizer.show_trajectories = False
visualizer.plot_3d_trajectories(save_path="no_trajectories.png")
```

### Parameter Analysis

```python
# Access generator parameters for custom analysis
print(f"Total subclasses: {generator.total_subclasses}")
print(f"Organ radii: {generator.organ_radii}")
print(f"Transition map: {generator.subclass_transition_map}")
```

### Import Path Note

The synthetic dataset generator is located in the IGNORE directory. Ensure the correct path is available:

```python
import sys
sys.path.append('/mnt/letitia/scratch/students/hhammoud/detr')
from synthetic_dataset_generator import SyntheticDatasetGenerator
```

## Interpreting Results

### Expected Behaviors

1. **Organ Persistence**: All organs should appear at all timepoints
2. **Energy Decay**: Average lesion energy should generally decrease over time
3. **Field Dynamics**: C1/C2 centers should move based on lesion positions
4. **Subclass Evolution**: Initial lesions may transition to evolved subclasses
5. **Parameter Consistency**: All parameters should be within specified ranges

### Warning Signs

1. **Disappearing Organs**: Indicates bug in persistence logic
2. **Energy Explosions**: Unrealistic energy gains may indicate field errors
3. **Empty Transitions**: Missing subclass transitions suggest configuration issues
4. **Parameter Outliers**: Values outside expected ranges (GAMMA_GAIN: 8.0-20.0, DELTA_DECAY: 0.1-0.35, Organ Radii: 0.3-0.7) indicate initialization problems
5. **No Dynamics**: Static behavior suggests deterministic step issues

### Performance Notes

- Visualization generation may take 30-60 seconds for complex plots
- Large datasets (>50 points) may result in cluttered 3D plots
- Animation generation requires additional memory for GIF creation
- All plots use non-interactive backend for headless compatibility

## File Outputs

All visualization methods support `save_path` parameter:
- **PNG files**: High-resolution static plots (300 DPI)
- **GIF files**: Animated sequences for temporal evolution
- **Automatic directory creation**: Output directories created as needed

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure matplotlib and seaborn are installed
2. **Display Issues**: Tool uses 'Agg' backend for headless environments
3. **Memory Warnings**: Large datasets may require more RAM
4. **Empty Plots**: Check that generator produced valid data
5. **Parameter Range Labels**: The validation plot may show outdated parameter ranges in labels; actual ranges are GAMMA_GAIN: 8.0-20.0, DELTA_DECAY: 0.1-0.35, Organ Radii: 0.3-0.7

### Debug Tips

```python
# Check generated data
coords_list, feats_list, labels_list = visualizer.coords_list, visualizer.feats_list, visualizer.labels_list
print(f"Timepoints: {len(coords_list)}")
for t in range(len(coords_list)):
    print(f"T{t}: {len(coords_list[t])} points")
```

This visualization tool provides comprehensive validation of the enhanced synthetic dataset generator, ensuring all complex dynamics are working as intended.
