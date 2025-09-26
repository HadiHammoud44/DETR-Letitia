#!/usr/bin/env bash
set -euo pipefail          # safer bash defaults

# 1. activate the virtual env
source /mnt/letitia/scratch/students/hhammoud/env/bin/activate

# 2. Define Timepoint and Modality
TP="TP2"
MODALITY="PT"
param_file="/mnt/letitia/scratch/students/hhammoud/param_pt.yaml"

# 3. Define input & output roots
IN_ROOT="/mnt/letitia/scratch/H_data/SAME/processed/${TP}/${MODALITY}"
OUT_ROOT="/mnt/letitia/scratch/H_data/SAME/features/${TP}/${MODALITY}"

# 4. Loop through every sub-directory (the “number” folders)
for dir in "$IN_ROOT"/*/; do
    number=$(basename "$dir")                         # just the folder name
    in_file="$IN_ROOT/$number/image.nii.gz"
    seg_dir="/mnt/letitia/scratch/H_data/SAME/segmented/${TP}/${MODALITY}/$number"
    out_dir="$OUT_ROOT/$number"

    # Create output directory if it doesn't exist
    mkdir -p "$out_dir"

    echo "Processing $number ..."
    
    # Loop through each segmented file in the directory
    for seg_file in "$seg_dir"/*; do
        if [[ -f "$seg_file" ]]; then
            echo "Processing segmentation file: $seg_file"

            # Ensure the segmentation file is a .nii.gz file
            if [[ ! "$seg_file" == *.nii.gz ]]; then
                echo "Skipping non-nii.gz file: $seg_file"
                continue
            fi

            base_name=$(basename "$seg_file")
            out_file="$out_dir/${base_name%.nii.gz}.csv"
            
            pyradiomics "$in_file" "$seg_file" -p "$param_file" -o "$out_file" -f csv
        fi
    done
done
echo "All cases processed."
