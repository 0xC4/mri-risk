from umcglib.utils import apply_parallel

import SimpleITK as sitk
import numpy as np


def preprocess_patient(patient_data: dict, configuration: dict):
    for scan_date in patient_data:
        scan_info = patient_data[scan_date]

        if "sitk" not in scan_info:
            continue
        
        if "numpy" not in scan_info:
            scan_info["numpy"] = {}
        sequences = list(scan_info["sitk"])
        for seq in sequences:
            img_s = scan_info["sitk"][seq]
            img_n = sitk.GetArrayFromImage(img_s).T

            norm_settings = configuration["normalization"][seq]
            if norm_settings["method"] == "znorm":
                img_n -= np.mean(img_n)
                img_n /= np.std(img_n)

            elif norm_settings["method"] == "divide":
                img_n /= norm_settings["divisor"]

            elif norm_settings["method"] == "clip":
                img_n = np.clip(img_n, norm_settings["min"], norm_settings["max"])

            scan_info["numpy"][seq] = img_n
    return patient_data


def preprocess_data(data: dict, configuration: dict, num_workers=8):
    patient_ids = sorted(data.keys())
    patient_data = [data[patient_id] for patient_id in patient_ids]

    # Load all patients in parallel
    patient_data = apply_parallel(
        patient_data, preprocess_patient, num_workers=num_workers, configuration=configuration
    )

    # Put it back in the dictionary format
    data = {pat_id: pat_dat for pat_id, pat_dat in zip(patient_ids, patient_data)}

    return data
