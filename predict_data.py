import json

import numpy as np
import pandas as pd

from umcglib.utils import tsprint

from lib.paths import get_scan_paths_nki
from lib.scans import load_scans_nki
from lib.preprocess import preprocess_data
from lib.model import SurvivalModel

if __name__ == "__main__":
    NUM_WORKERS = 1
    SCRATCH_ROOT = "./scratch/"
    DATASET_ROOT = "./predict_data/"
    DETECTION_MODEL_DIR = "models/detection/"
    SURVIVAL_MODEL_DIR = "models/survival/"
    TMPDIR = "./tmp/"
    NIFTI_OUTPUT_DIR = "./output_nifti/"
    MAX_WINDOW_SHAPE = 192, 192, 24
    SEQUENCES = ["t2w", "adc", "dwi"]
    AVAILABLE_PARAMETERS = ["psa", "psad", "prostate_volume", "patient_age"]
    TARGET = "gleason"
    TARGET_INTERVAL = 4 # years
    CLINICAL_MARKSHEET_PATH = "marksheet.csv"

    data = get_scan_paths_nki(DATASET_ROOT)

    tsprint("Adding clinical parameters..")
    marksheet_path = CLINICAL_MARKSHEET_PATH
    clinical_df = pd.read_csv(marksheet_path, sep=",")
    for idx, row in clinical_df.iterrows():
        patient_id = str(row.anon_id)
        try:
            scan_date = str(int(row.date_reformatted))
            print("SD", scan_date)
        except Exception as err:
            print(err)
            continue
        scan_date = f"{scan_date[:4]}-{scan_date[4:6]}-{scan_date[6:8]}"
        if patient_id not in data:
            print("Not found:", patient_id)
            continue
        if scan_date not in data[patient_id]:
            if not pd.isna(row.fixed_folder_number):
                fixed_date = str(int(row.fixed_folder_number))
                fixed_date = f"{fixed_date[:4]}-{fixed_date[4:6]}-{fixed_date[6:8]}"
                if fixed_date not in data[patient_id]:
                    continue
                
                data[patient_id][scan_date] = data[patient_id][fixed_date]
                data[patient_id][scan_date]["scan_date"] = scan_date
                
                tsprint("Updated scan reference from", scan_date, "to", fixed_date)
                
                del(data[patient_id][fixed_date])
            else:
                print("Not found:", patient_id, "Date:", scan_date)
                continue
                
        data[patient_id][scan_date]["clinical"] = {
            varname: row[varname] for varname in AVAILABLE_PARAMETERS
        }
        for varname in AVAILABLE_PARAMETERS:
            try:
                data[patient_id][scan_date]["clinical"][varname] = float(data[patient_id][scan_date]["clinical"][varname])
            except:
                data[patient_id][scan_date]["clinical"][varname] = None
        
        tsprint(data[patient_id][scan_date]["clinical"])
        
    tsprint("Loading scans..")
    data = load_scans_nki(
        data, num_workers=NUM_WORKERS, crop_shape=MAX_WINDOW_SHAPE, debug=False
    )

    preprocessing_config = {
        "normalization": {
            "t2w": {"method": "znorm"},
            "adc": {"method": "divide", "divisor": 4000.0},
            "dwi": {"method": "znorm"},
        },
    }
    print("\n## Preprocessing configuration ##:")
    print(json.dumps(preprocessing_config, sort_keys=True, indent=4))
    tsprint("Preprocessing data..")
    data = preprocess_data(
        data, preprocessing_config, num_workers=NUM_WORKERS
    )
    tsprint("Loading models..")
    model_folders = [
        "models/survival/fold_0/restart_25/",
        "models/survival/fold_1/restart_25/",
        "models/survival/fold_2/restart_25/",
        "models/survival/fold_3/restart_25/",
        "models/survival/fold_4/restart_25/"
    ]
    
                
    with open(f"results.csv", "w+") as f:
        f.write("fold_idx;patient_id;scan_date;calibrated_pred\n")
    
    for fold_idx, model_folder in enumerate(model_folders):
        tsprint(f"Loading model #{fold_idx}")
        
        model = SurvivalModel(
            SEQUENCES,
            TARGET,
            AVAILABLE_PARAMETERS,
            (192, 192, 24),
            detection_model_path=f"./models/detection/best_val_loss_fold_{fold_idx}.h5",
            l2_regularization=1e-3,
            instance_norm=False,
            optimizer="adam",
            learning_rate=1e-4,
            loss="binary_crossentropy",
            metrics=["AUC"],
            export_predictions_frequency=1,
            export_num_samples=200,
        )
        
        model.load_best_model(model_folder)
        
        for patient_id in data:
            patient_data = data[patient_id]
            for scan_date in patient_data:
                tsprint("Predicting:", patient_id, scan_date)
                scan_info = patient_data[scan_date]
                scans_n = np.stack([scan_info["numpy"][s] for s in SEQUENCES], axis=-1)
                scans_n = scans_n[None] # Add batch dimension
                print(scans_n.shape)
            
                clinvars_n = np.asarray([scan_info["clinical"][varname] for varname in AVAILABLE_PARAMETERS])
                clinvars_n = clinvars_n[None] # Add batch dimension
                print(clinvars_n.shape)
            
                target_interval_n = np.asarray([[TARGET_INTERVAL]])
                print(target_interval_n.shape)
                tsprint("Okidoke..")
                try:
                    pred_n = model.predict(scans_n, clinvars_n, target_interval_n)
                    tsprint("Predicted (normal):", pred_n)
                    pred_rev_n = model.predict(scans_n[:, ::-1], clinvars_n, target_interval_n)
                    tsprint("Predicted (flipped):", pred_rev_n)
                    pred_both_n = (pred_n + pred_rev_n) / 2
                    tsprint("Predicted (average):", pred_both_n)
                    calibrated_pred_n = model._apply_calibration(pred_both_n)[..., 0]
                    tsprint("Predicted (calibrated):", calibrated_pred_n)
                except:
                    tsprint("Skipping invalid..")
                    continue
                with open(f"results.csv", "a") as f:
                    f.write(f"{fold_idx};{patient_id};{scan_date};{calibrated_pred_n[0]:.3f}\n")

print("All done.")