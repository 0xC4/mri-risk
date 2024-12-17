# Deep learning-based risk prediction for prostate cancer progression
This repository contains the project code for the study: *Development and Validation of a Deep Learning Model Based on MRI and Clinical Characteristics to Predict Risk of Prostate Cancer Progression*

**Overview of contents:**
`lib/[paths,scans,preprocess,survival_data].py`: classes and functions for preprocessing paths and scans, to create a suitable dataset structure for further processing.
`lib/generator.py`: contains a batch generator that returns batches of (stratified) random observations during training.
`lib/[model,survival_model]`: model class and function for building the underlying tensorflow risk model

`make_dataset.py`: transforms a dataset of biparametric MRI scans + clinical variables + timepoints to the datastructure expected by the training routine. This involves transforming nifti images to .NPY for faster loading during training, preventing having to load everything into RAM.
