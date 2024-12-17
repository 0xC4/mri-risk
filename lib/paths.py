import datetime
from glob import glob
from os import path

import numpy as np
import SimpleITK as sitk


def get_scan_date(visit_dir):
    txt_file = path.join(visit_dir, "scan_date.txt")
    if path.exists(txt_file):
        for scan_date in open(txt_file):
            return scan_date.strip()
            
def get_scan_date_nki(visit_dir):
    visit_date = path.basename(path.dirname(visit_dir))
    visit_date = f"{visit_date[:4]}-{visit_date[4:6]}-{visit_date[6:8]}"
    return visit_date


def get_scan_paths(data_root):
    """ """
    data = {}  # patient_id->scan_date->sequence->scan_path
    for patient_dir in sorted(glob(path.join(data_root, "*/"))):
        patient_id = path.basename(path.dirname(patient_dir))
        for visit_dir in glob(path.join(patient_dir, "*/")):
            visit_id = path.basename(path.dirname(visit_dir))
            scan_date = get_scan_date(visit_dir)

            t2w_matches = glob(path.join(visit_dir, "t2_tra__*.nii.gz"))
            adc_matches = glob(path.join(visit_dir, "adc__*.nii.gz"))
            dwi_matches = glob(path.join(visit_dir, "high_b__*.nii.gz"))

            if not t2w_matches or not adc_matches or not dwi_matches:
                print(
                    f"[get_data] Missing scans for #{patient_id} ({scan_date})",
                    flush=True,
                )
                continue

            if patient_id not in data:
                data[patient_id] = {}

            # Redundancy in the information is just for ease of use later when
            # loading the scans in parallel
            data[patient_id][scan_date] = {
                "patient_id": patient_id,
                "scan_date": scan_date,
                "visit_id": visit_id,
                "visit_dir": visit_dir,
                "t2w": sorted(t2w_matches)[0],
                "adc": sorted(adc_matches)[0],
                "dwi": sorted(dwi_matches)[0],
            }

            gleason_matches = glob(path.join(visit_dir, "gleason.nii.gz"))
            pirads_matches = glob(path.join(visit_dir, "pirads.nii.gz"))

            for gleason_seg in gleason_matches:
                data[patient_id][scan_date]["gleason"] = gleason_seg
            for pirads_seg in pirads_matches:
                data[patient_id][scan_date]["pirads"] = pirads_seg

            registration_matches = glob(
                path.join(visit_dir, "registered", "*", "*.tfm")
            )
            for tfm_path in registration_matches:
                registration_type = path.basename(path.dirname(tfm_path))
                data[patient_id][scan_date][f"reg_{registration_type}"] = tfm_path

    return data
    
def get_scan_paths_nki(data_root):
    """ """
    data = {}  # patient_id->scan_date->sequence->scan_path
    for patient_dir in sorted(glob(path.join(data_root, "*/"))):
        patient_id = path.basename(path.dirname(patient_dir))
        for visit_dir in glob(path.join(patient_dir, "*/")):
            print("Processing", visit_dir) 
            visit_id = path.basename(path.dirname(visit_dir))
            scan_date = get_scan_date_nki(visit_dir)

            t2w_matches = glob(path.join(visit_dir, "t2w.nii.gz"))
            adc_matches = glob(path.join(visit_dir, "adc.nii.gz"))
            dwi_matches = glob(path.join(visit_dir, "dwi.nii.gz"))

            if not t2w_matches or not adc_matches or not dwi_matches:
                print(
                    f"[get_data] Missing scans for #{patient_id} ({scan_date})",
                    flush=True,
                )
                continue

            if patient_id not in data:
                data[patient_id] = {}

            # Redundancy in the information is just for ease of use later when
            # loading the scans in parallel
            data[patient_id][scan_date] = {
                "patient_id": patient_id,
                "scan_date": scan_date,
                "visit_id": visit_id,
                "visit_dir": visit_dir,
                "t2w": sorted(t2w_matches)[0],
                "adc": sorted(adc_matches)[0],
                "dwi": sorted(dwi_matches)[0],
            }

            gleason_matches = glob(path.join(visit_dir, "gleason.nii.gz"))
            pirads_matches = glob(path.join(visit_dir, "pirads.nii.gz"))

            for gleason_seg in gleason_matches:
                data[patient_id][scan_date]["gleason"] = gleason_seg
            for pirads_seg in pirads_matches:
                data[patient_id][scan_date]["pirads"] = pirads_seg

            registration_matches = glob(
                path.join(visit_dir, "registered", "*", "*.tfm")
            )
            for tfm_path in registration_matches:
                registration_type = path.basename(path.dirname(tfm_path))
                data[patient_id][scan_date][f"reg_{registration_type}"] = tfm_path

    return data
