# necessary imports
import os
import torch
import csv
import json
import ast

mapping_file = f"/mnt/letitia/scratch/H_data/SAME/idx_mapping.json"
with open(mapping_file, 'r') as f:
    mapping = json.load(f)

TP = 'TP0'
path = f"/mnt/letitia/scratch/H_data/SAME/features/{TP}/CT"
EXCLUDED_PATIENTS = [223, 125, 158, 14, 426, 404, 406, 415, 423, 460, 489]
EXCLUDED_ORGANS = ['prostate', 'gallbladder', 'adrenal_gland_left', 'adrenal_gland_right', 'thyroid_gland', 'urinary_bladder', 'spleen']
NB_FEATURES = 54


for patient_id in os.listdir(path):
    if int(patient_id) in EXCLUDED_PATIENTS:
        continue
    patient_path = os.path.join(path, patient_id)
    tensor = torch.tensor([])

    for ct_file in os.listdir(patient_path):
        if not ct_file.endswith('.csv') or ct_file[:-4] in EXCLUDED_ORGANS:
            continue 

        ct_path = os.path.join(patient_path, ct_file)
        with open(ct_path, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            if len(header) == 2:
                print(f"Warning: File {ct_path} has empty CT features. Inspect if not patient-129")
                continue
            else:
                l = next(reader)
                center_of_mass = list(ast.literal_eval(l[23]))
                ct_features = [float(x) for x in l[24:]]
        
        pt_path = ct_path.replace('CT', 'PT').replace('V', 'P')
        if not os.path.exists(pt_path):
            pt_features = [0 for _ in range(NB_FEATURES)]
        else:
            with open(pt_path, newline='') as f:
                reader = csv.reader(f)
                header = next(reader)
                if len(header) == 2:
                    print(f"Warning: File {pt_path} has empty PT features. Filling with zeros. Inspect it!")
                    pt_features = [0 for _ in range(NB_FEATURES)]
                else:
                    pt_features = [float(x) for x in next(reader)[24:]]

        foreground = [1 if ct_file.startswith('V') else 0] # 'superclass'
        subclass = [mapping[ct_file[:-4]]]  # map organ/lesion type to idx

        combined_features = torch.tensor(center_of_mass + ct_features + pt_features + foreground + subclass).unsqueeze(0)
        if tensor.numel() == 0:
            tensor = combined_features
        else:
            tensor = torch.cat((tensor, combined_features), dim=0)
        
    saving_path = path.replace('features', 'tensor_features').replace('/CT', '')
    os.makedirs(saving_path, exist_ok=True)
    torch.save(tensor, os.path.join(saving_path, f"{patient_id}.pt"))

    print(f"Processed patient {patient_id}, saved tensor of shape {tensor.shape} to folder {saving_path}")



        


