# Shared Utilities

Used throughout every stage of the pipeline. Add this folder to your MATLAB path before running
anything else (`addpath('06_shared_utilities')`).

- **`Shared.m`** -- a `classdef` of shared constants and static helper functions: windowing parameters
  (`FRAME_WINDOW`, `WINDOW_STEP`), the gesture class list, signal preprocessing (rectification +
  Butterworth low-pass envelope), spectrogram generation, the frame-labeling tolerance rule, and
  `NUM_VALID_TEST_MONTHS` (how many of Month 1--6 currently have consistent ground-truth annotation
  and are therefore included in evaluation). If you extend the dataset with additional annotated
  months, this is the one place to update.
- **`SpectrogramDatastore.m`** -- a custom MATLAB `datastore` subclass that reads the spectrogram
  sequence `.mat` files produced by `01_data_preparation/generateSpectrogramDataset.m`, used by the
  deep-backbone training scripts in `02_backbone_training/`.
