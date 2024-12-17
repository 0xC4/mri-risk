import SimpleITK as sitk

import umcglib.images as im
from umcglib.utils import apply_parallel, tsprint
from umcglib.registration import apply_transform


def load_patient(
    patient_data: dict,
    crop_shape=(160, 160, 16),
    spacing=(0.5, 0.5, 3.0),
    registration_method="baseline",
    debug=False,
):
    # Find the most recent scan
    num_scans = len(patient_data)
    scan_dates = sorted(patient_data, reverse=True)
    newest_scan_date = scan_dates[0]
    oldest_scan_date = scan_dates[-1]
    if debug:
        print(patient_data)
        print("Oldest:", oldest_scan_date, "Newest:", newest_scan_date)

    # Load the newest T2 first, so we can register other images to it
    reference_t2w = None
    reference_cropped_t2w = None
    for scan_date in scan_dates:
        scan_info = patient_data[scan_date]
        patient_id = scan_info["patient_id"]

        t2w_s = sitk.ReadImage(scan_info["t2w"], sitk.sitkFloat32)
        adc_s = sitk.ReadImage(scan_info["adc"], sitk.sitkFloat32)
        dwi_s = sitk.ReadImage(scan_info["dwi"], sitk.sitkFloat32)
        gleason_s = sitk.ReadImage(scan_info["gleason"], sitk.sitkFloat32)

        # Store the newest scan as the reference image for next images
        if scan_date == newest_scan_date:
            t2w_resampled_s = im.resample(
                t2w_s,
                min_shape=crop_shape,
                method=sitk.sitkLinear,
                new_spacing=spacing,
            )

            feasible_shape = [
                min(a, b) for a, b in zip(t2w_resampled_s.GetSize(), crop_shape)
            ]
            tsprint("Cropping image to:", feasible_shape)
            t2w_cropped_s = im.center_crop(t2w_resampled_s, feasible_shape)

            reference_t2w = t2w_s
            reference_cropped_t2w = t2w_cropped_s
        else:
            if not "reg_" + registration_method in scan_info:
                tsprint(
                    f"[load_patient][W] No '{registration_method}' transform "
                    + f"found for #{patient_id} ({scan_date})"
                )
                continue
            transform_path = scan_info[f"reg_{registration_method}"]

            # Read the previously determined registration transform
            registration_transform = sitk.ReadTransform(transform_path)

            # Apply it the scans, using the newest scans as the reference frame
            t2w_s = apply_transform(registration_transform, reference_t2w, t2w_s)
            adc_s = apply_transform(registration_transform, reference_t2w, adc_s)
            dwi_s = apply_transform(registration_transform, reference_t2w, dwi_s)
            gleason_s = apply_transform(
                registration_transform, reference_t2w, gleason_s
            )
            if debug:
                tsprint("[load_patient][D] Applied transform")

            # Preprocess the T2W first
            t2w_cropped_s = im.resample_to_reference(
                t2w_s, reference_cropped_t2w, interpolator=sitk.sitkLinear
            )

        # Then just resample the rest to it
        adc_cropped_s = im.resample_to_reference(
            adc_s, t2w_cropped_s, interpolator=sitk.sitkLinear
        )
        dwi_cropped_s = im.resample_to_reference(
            dwi_s, t2w_cropped_s, interpolator=sitk.sitkLinear
        )
        gleason_cropped_s = im.resample_to_reference(
            gleason_s, t2w_cropped_s, interpolator=sitk.sitkNearestNeighbor
        )

        scan_info["sitk"] = {
            "t2w": t2w_cropped_s,
            "adc": adc_cropped_s,
            "dwi": dwi_cropped_s,
            "gleason": gleason_cropped_s,
        }
        print(end=".", flush=True)

    if debug:
        for scan_date in patient_data:
            scan_info = patient_data[scan_date]
            if "sitk" not in scan_info:
                continue
            for sequence in scan_info["sitk"]:
                sitk.WriteImage(
                    scan_info["sitk"][sequence],
                    f"debug/{patient_id}_{scan_date}_{sequence}.nii.gz",
                )
    return patient_data


def load_scans(data: dict, num_workers=8, **kwargs):
    patient_ids = sorted(data.keys())
    patient_data = [data[patient_id] for patient_id in patient_ids]

    # Load all patients in parallel
    patient_data = apply_parallel(
        patient_data, load_patient, num_workers=num_workers, **kwargs
    )

    # Put it back in the dictionary format
    data = {pat_id: pat_dat for pat_id, pat_dat in zip(patient_ids, patient_data)}

    return data
    

def load_patient_nki(
    patient_data: dict,
    crop_shape=(160, 160, 16),
    spacing=(0.5, 0.5, 3.0),
    debug=False,
):
    # Find the most recent scan
    num_scans = len(patient_data)
    scan_dates = sorted(patient_data, reverse=True)
    newest_scan_date = scan_dates[0]
    oldest_scan_date = scan_dates[-1]
    if debug:
        print(patient_data)
        print("Oldest:", oldest_scan_date, "Newest:", newest_scan_date)

    # Load the newest T2 first, so we can register other images to it
    reference_t2w = None
    reference_cropped_t2w = None
    for scan_date in scan_dates:
        scan_info = patient_data[scan_date]
        patient_id = scan_info["patient_id"]

        t2w_s = sitk.ReadImage(scan_info["t2w"], sitk.sitkFloat32)
        adc_s = sitk.ReadImage(scan_info["adc"], sitk.sitkFloat32)
        dwi_s = sitk.ReadImage(scan_info["dwi"], sitk.sitkFloat32)

        t2w_resampled_s = im.resample(
            t2w_s,
            min_shape=crop_shape,
            method=sitk.sitkLinear,
            new_spacing=spacing,
        )

        feasible_shape = [
            min(a, b) for a, b in zip(t2w_resampled_s.GetSize(), crop_shape)
        ]
        tsprint("Cropping image to:", feasible_shape)
        t2w_cropped_s = im.center_crop(t2w_resampled_s, feasible_shape)

        reference_t2w = t2w_s
        reference_cropped_t2w = t2w_cropped_s

        # Then just resample the rest to it
        adc_cropped_s = im.resample_to_reference(
            adc_s, t2w_cropped_s, interpolator=sitk.sitkLinear
        )
        dwi_cropped_s = im.resample_to_reference(
            dwi_s, t2w_cropped_s, interpolator=sitk.sitkLinear
        )

        scan_info["sitk"] = {
            "t2w": t2w_cropped_s,
            "adc": adc_cropped_s,
            "dwi": dwi_cropped_s
        }
        print(end=".", flush=True)

    if debug:
        for scan_date in patient_data:
            scan_info = patient_data[scan_date]
            if "sitk" not in scan_info:
                continue
            for sequence in scan_info["sitk"]:
                sitk.WriteImage(
                    scan_info["sitk"][sequence],
                    f"debug/{patient_id}_{scan_date}_{sequence}.nii.gz",
                )
    return patient_data


def load_scans_nki(data: dict, num_workers=8, **kwargs):
    patient_ids = sorted(data.keys())
    patient_data = [data[patient_id] for patient_id in patient_ids]

    # Load all patients in parallel
    patient_data = apply_parallel(
        patient_data, load_patient_nki, num_workers=num_workers, **kwargs
    )

    # Put it back in the dictionary format
    data = {pat_id: pat_dat for pat_id, pat_dat in zip(patient_ids, patient_data)}

    return data
