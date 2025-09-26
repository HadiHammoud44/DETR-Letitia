# Synthetic Dataset: "Rule-Driven Points in a Box"

This README describes the generation of the synthetic dataset for temporal point cloud forecasting using DETR-like architectures. The dataset was designed to reflect the challenges encountered in LETITIA, specifically the existence of stable organs and **dynamic** lesions, while providing a controlled environment where the evolution is known and learnable, and does so across three timepoints (TP0 → TP1 → TP2). It is consistent with the data format expected by LETITIA-DETR training pipelines.

## Overview

**Point Types:**
- **Organs** (class 0): Persistent points with subclass-specific effect zones that influence nearby lesions
- **Lesions** (class 1): Dynamic points that can die/duplicate based on energy/instability, with initial→evolved subclass transitions

*Note:*
- *Points are named **organs** and **lesions** just to imitate the LETITIA dataset, although they do not have any biological meaning*
- *Subclass is a **semantic** label specifying the type of organ/lesion*

**Key Features:**
- Subclass-driven dynamics where each point's behavior is dictated by parameters unique to its subclass.
- Organ-lesion interactions via effect zones (energy reduction/instability boost)
- Lesion-lesion interactions via dynamic nutrient fields (C1/C2 centers)

## Dataset Format & Quick Start

**Structure:**  
`synthetic_dataset/TP{0,1,2}/{sample_id}.pt` - Each `.pt` file contains tensor `[N, 69]`

**Tensor Layout:**  
`[coords(3) + features(64) + class_label(1) + subclass_id(1)]`

```python
# Generate dataset
from synthetic_dataset_generator import SyntheticDatasetGenerator
generator = SyntheticDatasetGenerator(seed=0)
generator.generate_dataset(n_patients=10000, output_dir="synthetic_dataset")

# Load and analyze
import torch
data = torch.load("synthetic_dataset/TP0/0.pt")
coords, features, class_labels, subclass_ids = data[:,:3], data[:,3:67], data[:,67], data[:,68]
```

## Technical Specification

**Global Constants:** MAX_ORGANS=20, MAX_LESIONS=72, MIN_ORGANS=10, MIN_LESIONS=24

**Feature Vector (64-dim):** velocity(3) + energy(1) + instability(1) + MLP_extras(59)

### Subclass System
- **Organs:** Unique subclasses (0-19) with effect zones. Half are "gamma reducers" (energy↓), half are "instability boosters" (instability↑)
- **Lesions:** Split into initial (appear at TP0, can duplicate) and evolved (from duplication only) subclasses (20-91)
- **Parameters:** Each subclass has fixed behavioral parameters (thresholds, multipliers) across all samples

### Interaction Mechanisms
1. **Organ→Lesion:** Nearest organ affects lesion via radius-based zones
2. **Lesion→Lesion:** Dynamic nutrient field with C1/C2 centers (centroids of lesion subclass groups)

### Temporal Dynamics
**Position Update:** `coords += velocity*dt + noise`
**Lesion Energy:** `E = clamp(E + subclass_gamma_gain*organ_effect*nutrient_field - subclass_decay, 0, 1)`
**Lesion Fate:** 
- Death: `E < death_threshold` 
- Duplication: `E > dup_threshold AND instability > dup_threshold AND initial_subclass`
- On duplication: parent persists and a new child lesion is created with a predefined evolved subclass
