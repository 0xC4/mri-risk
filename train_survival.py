import os
from os import path
import sys

import numpy as np
import SimpleITK as sitk
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from lib.model import SurvivalModel
from lib.survival_data import load_prepared_dataset

NUM_WORKERS = os.cpu_count()
print(f"Found {NUM_WORKERS} CPUs, and using them all, of course :)")
NUM_FOLDS = 5 
current_fold = int(sys.argv[1])
current_restart = int(sys.argv[2])
WORK_DIR = f"./runs/fold_{current_fold}/restart_{current_restart}/"
PREPARED_DATASET_ROOT = (
    "./prepared_dataset/"
)
SEQUENCES = [
    "t2w",
    "adc",
    "dwi",
]
TARGET = "gleason"
CLINICAL_VARIABLES = [
    "psa", "psad", "prostate_volume", "patient_age",
    ]

model = SurvivalModel(
    SEQUENCES,
    TARGET,
    CLINICAL_VARIABLES,
    (192, 192, 24),
    detection_model_path=f"./detection_model/best_val_loss_fold_{current_fold}.h5",
    l2_regularization=1e-3,
    instance_norm=False,
    optimizer="adam",
    learning_rate=1e-4,
    loss="binary_crossentropy",
    metrics=["AUC"],
    export_predictions_frequency=1,
    export_num_samples=200,
)

training_folds = [i for i in range(NUM_FOLDS) if i != current_fold]
print(f"Current fold: fold {current_fold}")
print(f"Training on folds:", training_folds)

print("Loading prepared dataset:")
data = load_prepared_dataset(
    PREPARED_DATASET_ROOT, SEQUENCES, TARGET, CLINICAL_VARIABLES, training_folds
)
print("Done..")

print("Splitting training fold into training and validation..")
scans, clinvars, time_intervals, targets, visit_ids = data

num_total = len(scans)
num_train = int(num_total * 0.9)

all_idxs = list(range(num_total))
np.random.seed(seed=123*(1+current_restart))
np.random.shuffle(all_idxs)

train_idxs = all_idxs[:num_train]
valid_idxs = all_idxs[num_train:]

train_scans, train_clinvars, train_intervals, train_targets = (
    scans[train_idxs],
    clinvars[train_idxs],
    time_intervals[train_idxs],
    targets[train_idxs],
)

valid_scans, valid_clinvars, valid_intervals, valid_targets = (
    scans[valid_idxs],
    clinvars[valid_idxs],
    time_intervals[valid_idxs],
    targets[valid_idxs],
)
print(f"Done. Train scans: {len(train_scans)}, Valid scans: {len(valid_scans)}")

model.train(
    train_scans,
    train_clinvars,
    time_intervals[train_idxs],
    train_targets,
    valid_scans,
    valid_clinvars,
    time_intervals[valid_idxs],
    valid_targets,
    WORK_DIR,
    max_epochs=500,
    batch_size=20,
    early_stopping=20,
    monitor="val_loss",
    direction="min",
    calibrate_on_export=False
)

print("Loading test dataset:")
test_data = load_prepared_dataset(
    PREPARED_DATASET_ROOT, SEQUENCES, TARGET, CLINICAL_VARIABLES, [current_fold]
)
print("Done..")

test_prediction_dir = f"{WORK_DIR}/test_predictions"
os.makedirs(test_prediction_dir, exist_ok=True)

test_scans, test_clinvars, test_targets = test_data

predictions = model.predict(test_scans, test_clinvars)
calibrated_predictions = model._apply_calibration(predictions)

print("Exporting test images and predictions..")
for sample_idx in range(len(test_scans)):
    for seq_idx, sequence in enumerate(SEQUENCES):
        out_filename = f"{sample_idx:03d}_{sequence}.nii.gz"
        arr = test_scans[sample_idx, ..., seq_idx]
        arr = np.round(arr, 3)
        img = sitk.GetImageFromArray(arr.T)
        sitk.WriteImage(img, path.join(test_prediction_dir, out_filename))
        
    out_filename = f"{sample_idx:03d}_{TARGET}.nii.gz"
    arr = test_targets[sample_idx, ..., 0]
    seg = sitk.GetImageFromArray(arr.T)
    sitk.WriteImage(seg, path.join(test_prediction_dir, out_filename))
    
    pred = predictions[sample_idx, ..., 0]
    out_filename = f"{sample_idx:03d}_heatmap.nii.gz"
    arr = np.round(pred, 2)
    img = sitk.GetImageFromArray(arr.T)
    sitk.WriteImage(img, path.join(test_prediction_dir, out_filename))
    
    cal_pred = calibrated_predictions[sample_idx, ..., 0]
    out_filename = f"{sample_idx:03d}_heatmap_calibrated.nii.gz"
    arr = np.round(cal_pred, 2)
    img = sitk.GetImageFromArray(arr.T)
    sitk.WriteImage(img, path.join(test_prediction_dir, out_filename))
    
print("All done.")
