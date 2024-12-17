from os import path
import os
import pickle

import numpy as np
import tensorflow as tf
from focal_loss import BinaryFocalLoss
from betacal import BetaCalibration

from lib.survival_model import build_survival_model3
from umcglib.utils import tsprint

from lib.generator import balanced_datagenerator


def is_better(new_score, best_score, metric):
    """
    Short function that checks whether a new metric score is better than the
    previous best score.
    """
    if metric.replace("val_", "").lower() in ["auc", "accuracy", "pauc"]:
        return new_score > best_score
    else:
        return new_score < best_score


def worst_possible(metric):
    if metric.replace("val_", "").lower() in ["auc", "accuracy", "pauc"]:
        return 0.0
    else:
        return 999.0

# Models
class SurvivalModel:
    def __init__(
        self, sequences, target, clinical_parameters, input_shape, detection_model_path, **config
    ):
        self.sequences = sequences
        self.target = target
        self.clinical_parameters = clinical_parameters
        self.input_shape = input_shape
        self._detection_model_path = detection_model_path
        self._load_detection_model()
        self._model = None
        self._optimizer = None
        self._loss = None
        self._calibration_model = None
        self._metrics = []
        self.config = config

        self._initialize_model()
        self._model_compiled = False
        
    def _load_detection_model(self):
        tsprint("Loading detection model..")
        self._detection_model = tf.keras.models.load_model(
            self._detection_model_path, compile=False)
        self._detection_model.trainable = False
        self._detection_model.summary(line_length=120)
        tsprint("Loaded detection model succesfully.")

    def _initialize_model(self):
        tsprint("Initializing model..")
        c = self.config

        num_sequences = len(self.sequences)

        self._model = build_survival_model3(
            input_shape=self.input_shape + (num_sequences,),
            detection_model=self._detection_model,
            l2_regularization=c["l2_regularization"],
            instance_norm=c["instance_norm"],
            num_clinical_parameters=len(self.clinical_parameters),
        )

        tsprint("Model built:")
        self._model.summary(line_length=120)

    def _compile(self):
        c = self.config
        tsprint("Compiling model...")
        if c["optimizer"] == "adam":
            opt_func = tf.keras.optimizers.Adam
        elif c["optimizer"] == "rmsprop":
            opt_func = tf.keras.optimizers.RMSProp
        else:
            raise NotImplementedError("Unimplemented optimizer: " + c["optimizer"])
        self._optimizer = opt_func(c["learning_rate"])

        if c["loss"] == "focal_loss":
            self._loss = BinaryFocalLoss(pos_weight=c["pos_weight"], gamma=c["gamma"])
        else:
            self._loss = c["loss"]

        self._model.compile(
            optimizer=self._optimizer, loss=self._loss, metrics=c["metrics"]
        )
        self._model_compiled = True
        tsprint("Compiled model.")

    def load_best_model(self, work_dir: str):
        best_model_path = f"{work_dir}/models/best_val_AUC.h5"
        tsprint(f"Loading model from {best_model_path}..")
        self._model = tf.keras.models.load_model(best_model_path, compile=False)
        self._model.summary(line_length=120)
        self._model_compiled = False
        tsprint(f"Model loaded..")
        
        tsprint("Trying to load clinical variable means and stds")
        means_path = path.join(work_dir, "means.txt")
        stds_path = path.join(work_dir, "stds.txt")
        
        if path.exists(means_path):
            tsprint(f"Loading mean normalization parameters from {means_path}")
            tsprint(f"Loading SD normalization parameters from {stds_path}")
    
            self.clinical_means_train = np.asarray(
                [float(m.strip()) for m in open(means_path)]
            )
            self.clinical_stds_train = np.asarray(
                [float(m.strip()) for m in open(stds_path)]
            )
    
            tsprint("Loaded following means and SDs")
            print(self.clinical_means_train)
            print(self.clinical_stds_train)
        else:
            tsprint("No means and stds for clinical variables found in working directory.")
            tsprint("Skipping..")
            
        calibrator_path = f"{work_dir}/calibrator.pkl"
        if path.exists(calibrator_path):
            self.load_calibration_model(calibrator_path)
        else:
            tsprint(f"Not found: {calibrator_path}, skipping..")

    def _train_calibration(self, targets: np.ndarray, predictions: np.ndarray, save_as: str = None):
        targets_flat = targets.flatten()
        predictions_flat = predictions.flatten()
        
        tsprint("Fitting beta calibration model..")
        self._calibration_model = BetaCalibration(parameters="abm")
        self._calibration_model.fit(predictions_flat, targets_flat)
        
        if save_as is not None:
          tsprint(f"Saving calibration model to {save_as}..")
          with open(save_as, "wb+") as f:
              pickle.dump(self._calibration_model, f)
        tsprint("Done.")
        
    def load_calibration_model(self, pkl_path: str):
        tsprint("Loading calibration model from", pkl_path)
        with open(pkl_path, "rb") as f:
            self._calibration_model = pickle.load(f)
        tsprint("Done.")            
        
    def _apply_calibration(self, predictions: np.ndarray):
        # Ensure that the calibration model is initialized
        if self._calibration_model is None:
            tsprint("[W] Calibration model not initialized, returning original predictions.")
            return predictions
            
        tsprint("Calibrating model predictions..")
        calibrated_predictions = [
            self._calibration_model.predict(pred.flatten()).reshape(pred.shape) for pred in predictions
        ]
        calibrated_predictions = np.asarray(calibrated_predictions, dtype=np.float32)
        tsprint("Done.")
        return calibrated_predictions

    def train(
        self,
        train_scans: np.ndarray,
        train_clin_params: np.ndarray,
        train_intervals: np.ndarray,
        train_targets: np.ndarray,
        valid_scans: np.ndarray,
        valid_clin_params: np.ndarray,
        valid_intervals: np.ndarray,
        valid_targets: np.ndarray,
        work_dir,
        max_epochs=500,
        batch_size=8,
        early_stopping=20,
        monitor="val_loss",
        direction="min",
        calibrate_on_export=True,
        **kwargs,
    ):
        c = self.config
        prediction_dir = path.join(work_dir, "predictions")
        model_dir = path.join(work_dir, "models")
        os.makedirs(prediction_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        log_file = path.join(work_dir, "train_log.csv")
        num_train_samples = len(train_scans)

        best_scores = {}

        if not self._model_compiled:
            self._compile()

        if len(self.clinical_parameters) > 0:
            tsprint("Determining training means to impute missing information..")
            self.clinical_means_train = np.nanmean(train_clin_params, axis=0)
            self.clinical_stds_train = np.nanstd(train_clin_params, axis=0)
            tsprint("Means:", self.clinical_means_train)
            tsprint("STDs:", self.clinical_stds_train)
            tsprint("Saving means and STDs to file")
            with open(path.join(work_dir, "means.txt"), "w+") as f:
                f.writelines([str(m) + "\n" for m in self.clinical_means_train])
            with open(path.join(work_dir, "stds.txt"), "w+") as f:
                f.writelines([str(m) + "\n" for m in self.clinical_stds_train])
            
            for col in range(train_clin_params.shape[1]):
                tsprint(f"Found missing: {np.sum(np.isnan(train_clin_params[:, col]))}")
                train_clin_params[np.isnan(train_clin_params[:, col]), col] = self.clinical_means_train[col]
                valid_clin_params[np.isnan(valid_clin_params[:, col]), col] = self.clinical_means_train[col]
                tsprint(f"After imputation: {np.sum(np.isnan(train_clin_params[:, col]))}")
                
            tsprint("Z-norming clinical parameters using train means and stds..")
            train_clin_params = (train_clin_params - self.clinical_means_train) / self.clinical_stds_train
            valid_clin_params = (valid_clin_params - self.clinical_means_train) / self.clinical_stds_train
            
            valid_data = (valid_scans, valid_clin_params, valid_intervals), valid_targets
        else:
            valid_data = (valid_scans, valid_intervals), valid_targets
        
        train_generator = balanced_datagenerator(
            train_scans,
            train_targets,
            train_clin_params,
            train_intervals,
            batch_size,
            shuffle=True,
            crop_shape=self.input_shape,
        )
        
        training_completed = False
        export_frequency = c["export_predictions_frequency"]
        for epoch_num in range(max_epochs):
            tsprint(f"Starting epoch {epoch_num:03d}")
            epoch = self._model.fit(
                train_generator,
                #train_inputs, train_targets,
                validation_data=valid_data,
                #validation_batch_size=batch_size,
                epochs=1,
                steps_per_epoch=num_train_samples//batch_size
                #batch_size=batch_size,
                #steps_per_epoch=1 # DEBUG
            )

            if epoch_num == 0:
                tsprint("Creating log file at", log_file)
                metric_keys = sorted(epoch.history.keys())
                header = ";".join(["epoch"] + metric_keys) + "\n"
                with open(log_file, "w+") as f:
                    f.write(header)

            with open(log_file, "a") as f:
                f.write(f"{epoch_num};")
                f.write(";".join([str(epoch.history[m][0]) for m in metric_keys]))
                f.write("\n")

            # Check if validation performance improved
            for metric in metric_keys:
                if metric not in best_scores:
                    best_scores[metric] = {"epoch": -1, "score": worst_possible(metric)}

                if "val_" not in metric:
                    continue

                best_score = best_scores[metric]["score"]
                best_epoch = best_scores[metric]["epoch"]
                score = epoch.history[metric][0]
                if is_better(score, best_score, metric):
                    tsprint(
                        f"New best {metric}: {score:0.5f};"
                        + f" Previous best: {best_score:0.5f} (epoch {best_epoch})"
                    )
                    best_scores[metric]["score"] = score
                    best_scores[metric]["epoch"] = epoch_num

                    model_path = path.join(model_dir, f"best_{metric}.h5")
                    tsprint("Saving model to", model_path)
                    self._model.save(model_path, overwrite=True)
                else:
                    tsprint(f"{metric} did not improve from {best_score:0.5f} (epoch {best_epoch})")
                    if monitor == metric:
                        if epoch_num - best_epoch >= early_stopping:
                            tsprint(f"Early stopping after {early_stopping} epochs without improvement.")
                            training_completed = True

            if epoch_num % export_frequency == 0:
                tsprint("Exporting predictions")
                predictions = self._model.predict(valid_data[0], batch_size=1)
                tsprint("#### PREDICTIONS ####")
                for pred, target in zip(predictions, valid_targets):
                    print(f"Pred: {pred}; Target: {target}")
                    
                
                tsprint("Exporting progression curve")
                NUM_TIMEPOINTS = 11
                for sample_idx in range(20):
                    tsprint("##### CASE", sample_idx, "######")
                    scans = np.stack([valid_data[0][0][sample_idx]] * NUM_TIMEPOINTS)
                    clinvars = np.stack([valid_data[0][1][sample_idx]] * NUM_TIMEPOINTS)
                    intervals = np.linspace(0, 5, NUM_TIMEPOINTS)
                    target = [valid_data[1][sample_idx]]
                    predictions = self._model.predict((scans, clinvars, intervals), batch_size=1)
                
                    tsprint("#### PREDICTIONS ####")
                    for t, pred in zip(intervals, predictions):
                        print(f"T: {t}; Pred: {pred}; Target: {target}")
                    print(flush=True)  
            
            if training_completed:
                break
        
        tsprint("Training finished, reloading best model for calibration..")
        self.load_best_model(work_dir)
        predictions = self._model.predict(valid_data[0], batch_size=1)
        
        tsprint("All done.")

    def predict(self, scans, clinical_variables=None, test_intervals=None, calibrated=False):
        tsprint("Preparing to predict..")
        test_data = scans
            
        if clinical_variables is not None and clinical_variables.shape[1] > 0:
            tsprint("Mean-imputing missing clinical parameters..")
            for col in range(clinical_variables.shape[1]):
                tsprint(f"Found missing: {np.sum(np.isnan(clinical_variables[:, col]))}")
                clinical_variables[np.isnan(clinical_variables[:, col]), col] = self.clinical_means_train[col]
                tsprint(f"After imputation: {np.sum(np.isnan(clinical_variables[:, col]))}")
                
            tsprint("Z-norming clinical parameters using train means and stds..")
            clinical_variables = (clinical_variables - self.clinical_means_train) / self.clinical_stds_train
            test_data = (scans, clinical_variables, test_intervals)
        
        tsprint("Predicting..")
        predictions = self._model.predict(test_data, batch_size=1)
        
        if calibrated:
            tsprint("Calibrating predictions..")
            predictions = self._apply_calibration(predictions)
            
        tsprint("Done.")
        
        return predictions
