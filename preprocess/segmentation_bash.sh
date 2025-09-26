#!/usr/bin/env bash
set -euo pipefail          # safer bash defaults

# 1. activate the virtual env
source /mnt/letitia/scratch/students/hhammoud/env/bin/activate   

# 2. Define input & output roots
TP="TP0"  
CT_ROOT="/mnt/letitia/scratch/H_data/SAME/processed/${TP}/CT"
OUT_ROOT="/mnt/letitia/scratch/H_data/SAME/testt/${TP}/CT"

# 3. Loop through every sub-directory (the “number” folders)
for dir in "$CT_ROOT"/*/; do
    number=$(basename "$dir")                         # just the folder name
    in_file="$CT_ROOT/$number/image.nii.gz"
    out_dir="$OUT_ROOT/$number"

    echo "Processing $number ..."
    TotalSegmentator -i "$in_file" -o "$out_dir" --device gpu \
    --roi_subset brain spinal_cord thyroid_gland trachea lung_upper_lobe_left lung_upper_lobe_right lung_middle_lobe_right \
    lung_lower_lobe_left lung_lower_lobe_right adrenal_gland_left adrenal_gland_right spleen liver gallbladder kidney_left kidney_right pancreas 
done

echo "All cases processed."
