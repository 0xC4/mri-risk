import os
import sys
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from umcglib.utils import tsprint
from umcglib import images as im

from lib.paths import get_scan_paths
from lib.scans import load_scans
from lib.preprocess import preprocess_data
from lib.survival_data import prepare_data

if __name__ == "__main__":
    NUM_WORKERS = os.cpu_count()
    PROJECT_ROOT = "/home2/survival"
    SCRATCH_ROOT = "/scratch/survival"
    DATASET_ROOT = "/scratch/datasets/follow-up"
    MAX_WINDOW_SHAPE = 192, 192, 24
    SEQUENCES = ["t2w", "adc", "dwi"]
    AVAILABLE_PARAMETERS = ["psa", "psad", "prostate_volume", "patient_age"]

    data = get_scan_paths(DATASET_ROOT)

    current_fold = int(sys.argv[1])
    output_dataset_root = f"./prepared_dataset/fold_{current_fold}/"
    os.makedirs(output_dataset_root, exist_ok=True)

    patient_ids = [
        l.strip()
        for l in open(f"{PROJECT_ROOT}/data/splits/fold_{current_fold}_patient_ids.txt")
    ]
    fold_data = {pat_id: data[pat_id] for pat_id in patient_ids}

    tsprint("Adding clinical parameters..")
    marksheet_path = f"{PROJECT_ROOT}/data/marksheet.csv"
    clinical_df = pd.read_csv(marksheet_path, sep=";")
    for idx, row in clinical_df.iterrows():
        patient_id = str(row.patient_id)
        scan_date = row.anon_mri_date.strip()
        if patient_id not in data:
            print("Not found:", patient_id)
            continue
        data[patient_id][scan_date]["clinical"] = {
            varname: row[varname] for varname in AVAILABLE_PARAMETERS
        }

    tsprint("Clinical varnames:", AVAILABLE_PARAMETERS)

    tsprint("Loading scans..")
    fold_data = load_scans(
        fold_data, num_workers=NUM_WORKERS, crop_shape=MAX_WINDOW_SHAPE, debug=False
    )

    preprocessing_config = {
        "normalization": {
            "t2w": {"method": "znorm"},
            "adc": {"method": "divide", "divisor": 4000.0},
            "dwi": {"method": "znorm"},
            "gleason": {"method": "clip", "min": 0.0, "max": 1.0},
        },
    }
    print("\n## Preprocessing configuration ##:")
    print(json.dumps(preprocessing_config, sort_keys=True, indent=4))
    tsprint("Preprocessing data..")
    fold_data = preprocess_data(
        fold_data, preprocessing_config, num_workers=NUM_WORKERS
    )
    tsprint("Preparing sequential numpy dataset")
    sequential_samples = prepare_data(
        fold_data,
        SEQUENCES,
        "gleason",
        AVAILABLE_PARAMETERS,
        num_workers=128,
    )

    tsprint(f"Exporting .npy's to {output_dataset_root}")
    with open(output_dataset_root+"/clinvar_names.txt", 'w+') as f:
        for v in AVAILABLE_PARAMETERS:
            f.write(v+"\n")

    for (scans, target, patient_id, current_date, prior_date, clinical_vars, years_between) in tqdm(sequential_samples):
        prefix = f"{patient_id}_{prior_date}_{current_date}"
        for seq_idx, seq in enumerate(SEQUENCES):
            scan_n = scans[..., seq_idx]
            nifti_path = f"{output_dataset_root}/{prefix}_{seq}.nii.gz"
            tsprint("Saving :", nifti_path)
            im.to_sitk(scan_n, save_as=nifti_path)
        # Export target
        target_path = f"{output_dataset_root}/{prefix}_gleason.txt"
        tsprint("Saving :", target_path)
        with open(target_path, "w+") as f:
            f.write(str(target) + "\n")
        # Export interval
        interval_path = f"{output_dataset_root}/{prefix}_years_between.txt"
        tsprint("Saving :", interval_path)
        with open(interval_path, "w+") as f:
            f.write(str(years_between) + "\n")
        # Export clinical variables
        npy_path = f"{output_dataset_root}/{prefix}_clinvars.npy"
        np.save(npy_path, clinical_vars)
        
