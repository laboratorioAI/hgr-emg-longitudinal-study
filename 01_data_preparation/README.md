# Stage 0 -- Data Preparation

Turns raw per-participant sEMG recordings into the two cached datasets every later stage builds on.

## Scripts, in the order you would typically run them

1. **`addGroundTruthFromStudents.m`** and **`groundTruthIndexDatasetMap.m`** -- one-time scripts that
   attach a manually-segmented `groundTruthIndex = [start, end]` sample-index pair to each repetition,
   for the months whose gesture boundaries were annotated by student collaborators rather than being
   present in the raw capture. Student names in the original lab records were replaced with generic
   aliases (`Estudiante_A`..`Estudiante_H`) before publishing this script; the mapping from alias to
   which month/participant range each alias covers is preserved exactly. **Fill in the
   `<RUTA_...>` path placeholders at the top of each file before running.** Skip these two scripts
   entirely if your copy of the dataset already has `groundTruthIndex` populated for every month.

2. **`generateSpectrogramDataset.m`** -- for the deep-learning path. Converts each windowed frame of
   preprocessed EMG into an STFT spectrogram tensor, splits Month 0 80/20 into training/validation *by
   repetition* (not by participant -- every participant appears on both sides), and writes everything
   to `DatastoresTran/<split>/<gesture>/*.mat`. This is what `02_backbone_training/` consumes.

3. **`generateStatFeatureDataset.m`** -- for the classical-ML path. Same windowing, same split logic,
   same random seed as step 2 (so the two paths see the identical calibration/evaluation data), but
   computes a 32-d hand-crafted statistical feature vector (mean absolute value, RMS, standard
   deviation, energy, per channel) instead of a spectrogram. Writes to
   `DatastoresStatFeatures/<split>/<gesture>/*.mat`. This is what `03_classical_baselines/` consumes.

Both dataset-generation scripts must be pointed at your own local copy of the raw data before running
-- see the path placeholders at the top of each file and the top-level [README](../README.md#directory-layout-you-need-to-provide)
for the expected raw-data layout.

## Reproducibility note

Both `generateSpectrogramDataset.m` and `generateStatFeatureDataset.m` call `rng(9)` before the Month-0
train/validation split, so re-running either from a clean output directory reproduces the identical
split. **Do not re-run either script into a non-empty output directory** -- neither script clears its
own output folder first, so a second run with different logic (e.g. after a bug fix) will silently
leave old files mixed in with new ones. Delete `DatastoresTran/` / `DatastoresStatFeatures/` completely
before regenerating.
