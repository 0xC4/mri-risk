from os import path

from glob import glob

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from umcglib.utils import tsprint, apply_parallel

from lib.clinical_information import get_clinical_information


def get_observations_from_patient(
    patient_data: dict,
    sequences: list,
    target: str,
    sequential_type: str,
    clinical_parameters: list = None,
    differential=False,
):
    """
    Takes a single patient with 0 or more visits, and extracts
    0 or more preprocessed sequential or single scan observations for
    training a model.

    `sequential_type`:
        "vp+c": variable prior + current (most recent scan is always current,
            multiple prior candidates)
        "p+vc": prior + variable current (prior is always first scan,
            multiple current candidates)
        "fp+lc": first prior + last current (oldest and newest scans)
        "fp+fc": first prior + subsequent current scan
        "f": first scan
        "l": last scan
        "as": all scans extracted as single scans
    """

    scan_dates = sorted(patient_data)
    num_visits = len(scan_dates)

    observations = []

    # First do options for which a single scan is sufficient
    if num_visits < 1:
        return []

    oldest_scan = scan_dates[0]
    newest_scan = scan_dates[-1]

    if sequential_type == "a":
        for current_date in scan_dates:
            current_scan = patient_data[current_date]
            current_arrays = [current_scan["numpy"][seq] for seq in sequences]
            current_combined_n = np.stack(current_arrays, axis=-1)

            # Use the target for the current scan as the overall target
            target_n = current_scan["numpy"][target]

            # Expand it with a channel axis
            target_n = target_n[..., None]

            observations.append(
                (
                    current_combined_n,
                    target_n,
                    current_scan["patient_id"],
                    current_date,
                    None,
                )
            )
        return observations

    elif sequential_type == "f":
        raise NotImplementedError()
    elif sequential_type == "l":
        raise NotImplementedError()

    # Now do options for which at least two scans are required
    if num_visits < 2:
        return []

    elif sequential_type == "c":
        current_dates = scan_dates[1:]
        for current_date in current_dates:
            current_scan = patient_data[current_date]
            current_arrays = [current_scan["numpy"][seq] for seq in sequences]
            current_combined_n = np.stack(current_arrays, axis=-1)

            # Use the target for the current scan as the overall target
            target_n = current_scan["numpy"][target]

            # Expand it with a channel axis
            target_n = target_n[..., None]

            # Add optional clinical variables
            if clinical_parameters is None:
                clinical_n = None
            else:
                clinical_n = get_clinical_information(
                    current_dict=current_scan,
                    prior_dict=None,
                    clinical_vars=clinical_parameters,
                    differential=differential,
                )

            observations.append(
                (
                    current_combined_n,
                    target_n,
                    current_scan["patient_id"],
                    current_date,
                    None,
                    clinical_n,
                )
            )
        return observations

    if sequential_type == "p+vc":
        prior_date = oldest_scan
        prior_scan = patient_data[prior_date]

        # Concatenate the sequences along the channel axis (-1)
        prior_arrays = [prior_scan["numpy"][seq] for seq in sequences]
        prior_combined_n = np.stack(prior_arrays, axis=-1)

        current_dates = scan_dates[1:]
        for current_date in current_dates:
            current_scan = patient_data[current_date]
            current_arrays = [current_scan["numpy"][seq] for seq in sequences]
            current_combined_n = np.stack(current_arrays, axis=-1)

            # Combine prior and current along the (existing) channel axis
            paired_scans_n = np.concatenate(
                [prior_combined_n, current_combined_n], axis=-1
            )

            # Use the target for the current scan as the overall target
            target_n = current_scan["numpy"][target]

            # Expand it with a channel axis
            target_n = target_n[..., None]

            # Add optional clinical variables
            if clinical_parameters is None:
                clinical_n = None
            else:
                clinical_n = get_clinical_information(
                    current_dict=current_scan,
                    prior_dict=prior_scan,
                    clinical_vars=clinical_parameters,
                    differential=differential,
                )

            observations.append(
                (
                    paired_scans_n,
                    target_n,
                    current_scan["patient_id"],
                    current_date,
                    prior_date,
                    clinical_n,
                )
            )
        return observations

    elif sequential_type == "f":
        raise NotImplementedError()
    elif sequential_type == "l":
        raise NotImplementedError()
    else:
        raise ValueError("Unknown data selector: {}".format(sequential_type))


def prepare_data(
    data: dict,
    sequences: list,
    target: str,
    sequential_type: str,
    clinical_parameters: list = None,
    differential=False,
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
            sequential_type=sequential_type,
            clinical_parameters=clinical_parameters,
            differential=differential,
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

    target_s = sitk.ReadImage(f"{prefix}_{target}.nii.gz", sitk.sitkFloat32)
    target_n = sitk.GetArrayFromImage(target_s).T[..., None]

    all_clinvars = np.load(f"{prefix}_clinvars.npy")
    clinvar_dict = {
        varname: value for varname, value in zip(clinvar_header, all_clinvars)
    }

    clinvars_n = np.asarray([clinvar_dict[varname] for varname in clinical_variables])
    print(end=".", flush=True)
    return scans_n, clinvars_n, target_n


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
    pat_prefixes = ["_".join(fp.split("_")[:-2]) for fp in paths]
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
    Y = np.asarray([d[2] for d in dataset], dtype=np.float32)

    tsprint(X_scan.shape, X_clin.shape, Y.shape)

    return X_scan, X_clin, Y, visit_ids
