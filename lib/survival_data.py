from os import path
from datetime import datetime
from glob import glob

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from umcglib.utils import tsprint, apply_parallel


def get_clinical_information(prior_dict, clinical_vars):
    prior_vars = [prior_dict.get("clinical", {}).get(v) for v in clinical_vars]
    return np.asarray(prior_vars, dtype=np.float32)


def get_observations_from_patient(
    patient_data: dict,
    sequences: list,
    target: str,
    clinical_parameters: list = None
):
    """
    Takes a single patient with 0 or more visits, and extracts
    0 or more preprocessed sequential or single scan observations for
    training a model.
    """

    scan_dates = sorted(patient_data)
    num_visits = len(scan_dates)

    observations = []

    # If less than 2 timepoints available skiponly 
    if num_visits <= 1:
        return []
    
    for prior_idx, prior_date in enumerate(scan_dates[:-1]):
        prior_scan = patient_data[prior_date]
        prior_arrays = [prior_scan["numpy"][seq] for seq in sequences]
        prior_combined_n = np.stack(prior_arrays, axis=-1)

        # Add optional clinical variables
        if clinical_parameters is None:
            clinical_n = None
        else:
            clinical_n = get_clinical_information(prior_scan, clinical_parameters)
            
        for current_date in scan_dates[prior_idx:]:
            # Get current scan
            current_scan = patient_data[current_date]
    
            # Determine if the target map was positive for csPCa
            target_label = current_scan["numpy"][target].any() * 1
                
            # Add time difference
            prior_datetime = datetime.strptime(prior_date, "%Y-%m-%d")
            current_datetime = datetime.strptime(current_date, "%Y-%m-%d")
            years_between = (current_datetime - prior_datetime).days / 356.0
    
            observations.append(
                (
                    prior_combined_n,
                    target_label,
                    current_scan["patient_id"],
                    current_date,
                    prior_date,
                    clinical_n,
                    years_between,
                )
            )
    return observations


def prepare_data(
    data: dict,
    sequences: list,
    target: str,
    clinical_parameters: list = None,
    num_workers=128,
):
    # Convert from list to dictionary to allow parallelization
    patient_ids = sorted(data)
    patient_data = [data[pat_id] for pat_id in patient_ids]

    observations_per_patient = []
    patients_per_iteration = len(patient_data) // 10 + 1
    for i in tqdm(range(10)):
        start = i * patients_per_iteration
        end = (i + 1) * patients_per_iteration
        observations_per_patient += apply_parallel(
            patient_data[start:end],
            get_observations_from_patient,
            sequences=sequences,
            target=target,
            clinical_parameters=clinical_parameters,
            num_workers=num_workers,
        )

    all_observations = sum(observations_per_patient, [])
    return all_observations


def load_single_observation(
    prefix, sequences, target, clinical_variables, clinvar_header
):
    scans_s = [
        sitk.ReadImage(f"{prefix}_{seq}.nii.gz", sitk.sitkFloat32) for seq in sequences
    ]
    scans_n = np.stack([sitk.GetArrayFromImage(s).T for s in scans_s], axis=-1)

    target_label = [int(l.strip()) for l in open(f"{prefix}_{target}.txt")][0]
    years_between = [float(l.strip()) for l in open(f"{prefix}_years_between.txt")][0]

    all_clinvars = np.load(f"{prefix}_clinvars.npy")
    clinvar_dict = {
        varname: value for varname, value in zip(clinvar_header, all_clinvars)
    }

    clinvars_n = np.asarray([clinvar_dict[varname] for varname in clinical_variables])
    print(end=".", flush=True)
    return scans_n, clinvars_n, years_between, target_label


def load_prepared_dataset(
    root, sequences, target, clinical_variables, folds, num_workers=64
):
    first_sequence = sequences[0]
    paths = sum(
        [sorted(glob(f"{root}/fold_{fold}/*_{first_sequence}.nii.gz")) for fold in folds], []
    )
    visit_ids = [path.basename(p)[:16] for p in paths]
    clinvar_header_path = sum(
        [glob(f"{root}/fold_{fold}/clinvar_names.txt") for fold in folds], []
    )[0]
    clinvar_header = [l.strip() for l in open(clinvar_header_path)]
    pat_prefixes = ["_".join(fp.split("_")[:-1]) for fp in paths]
    dataset = apply_parallel(
        pat_prefixes,
        load_single_observation,
        sequences=sequences,
        target=target,
        clinical_variables=clinical_variables,
        clinvar_header=clinvar_header,
        num_workers=num_workers,
    )

    X_scan = np.asarray([d[0] for d in dataset], dtype=np.float32)
    X_clin = np.asarray([d[1] for d in dataset], dtype=np.float32)
    X_time = np.asarray([d[2] for d in dataset], dtype=np.float32)
    Y = np.asarray([d[3] for d in dataset], dtype=np.float32)

    tsprint(X_scan.shape, X_clin.shape, X_time.shape, Y.shape)

    return X_scan, X_clin, X_time, Y, visit_ids
