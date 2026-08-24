# Longitudinal Drift in sEMG Hand-Gesture Recognition

Code accompanying the study *"Longitudinal Drift in sEMG Hand-Gesture Recognition: Robustness Across
Decision Paradigms, Architectures, and Individuals"* (submitted to ICASSP 2027).

This repository contains the full MATLAB pipeline used to produce every number, table, and figure in
the paper: a six-month longitudinal evaluation of hand-gesture recognition from surface
electromyography (sEMG), comparing a supervised Softmax classifier against a LinUCB contextual
bandit on a frozen CNN-Transformer-encoder representation (three backbone capacities), two classical
machine-learning baselines (SVM, k-NN) as an architectural-robustness check, and a personalized,
per-participant re-analysis.

## Data availability

**The raw sEMG dataset is not included in this repository.** It is longitudinal data from 19 human
participants recorded over seven sessions; the informed consent obtained from participants supports
its use for the research program of the Artificial Intelligence and Computer Vision Research Lab at
Escuela Politécnica Nacional, and does not currently extend to unrestricted public redistribution
(broadening this consent is in progress), and the dataset remains under active use by other ongoing
studies in the same laboratory. **The dataset is available from the authors upon reasonable request** --
see the paper's *Data and Code Availability* section, or open an issue / contact the corresponding
author (see [`07_figures_and_paper/paper/ICASSP2027-paper.tex`](07_figures_and_paper/paper/ICASSP2027-paper.tex)).

Everything in this repository is written to work against your own local copy of the dataset once
obtained, using the same directory layout the original study used (see [Directory layout you need to
provide](#directory-layout-you-need-to-provide) below). A handful of scripts that touch raw-data
preparation contain path placeholders (`<RUTA_DATASET_CRUDO>`, etc.) that you fill in with your own
paths; this is called out at the top of each such file.

## Repository structure

The code is organized by pipeline stage, not by when each script was written. Run stages in the
numeric order below; within a stage, scripts are meant to be run in the order they are described.

| Folder | Stage | What it does |
|---|---|---|
| [`01_data_preparation/`](01_data_preparation/) | 0 | Turns raw per-user sEMG recordings into two parallel cached datasets: STFT spectrogram sequences (for the deep backbone) and hand-crafted statistical features (for SVM/k-NN). Also includes the one-time scripts that attached manually-segmented ground-truth gesture boundaries to the raw recordings for the months that needed it. |
| [`02_backbone_training/`](02_backbone_training/) | 1 | Trains the CNN-Transformer-encoder backbone (three capacities: large/medium/small) and the TCN alternative-architecture backbone on Month 0, then freezes it and caches its 128-d (or 32-d/64-d, depending on capacity) representation for every later month. |
| [`03_classical_baselines/`](03_classical_baselines/) | 1 (parallel to Stage 1) | Builds the 32-d hand-crafted feature cache (MAV/RMS/SD/energy) and trains the SVM, k-NN, and no-CNN ANN classical baselines on it, including the fixed-hyperparameter robustness variants. |
| [`04_decision_heads/`](04_decision_heads/) | 2 | Trains the two decision mechanisms compared in the paper -- a supervised Softmax head and a LinUCB contextual bandit (frozen and adaptive variants) -- on top of the frozen backbone representation from Stage 1, and compares them across backbone capacities and months. |
| [`05_covariate_shift/`](05_covariate_shift/) | 2 (independent of Stage 2) | Tests the frozen 128-d representation itself for covariate shift between Month 0 and each later month, using MMD² and a domain-classifier A-distance, independently of any decision head's accuracy. |
| [`06_personalized_models/`](06_personalized_models/) | 3 | Repeats the evaluation with one independent model per participant (instead of one pooled population model) for SVM, k-NN, and Softmax, to test whether population-averaged accuracy curves represent individual participants. |
| [`07_figures_and_paper/`](07_figures_and_paper/) | 4 | Generates the English-language figures used in the ICASSP paper, and contains the paper's LaTeX source itself (`paper/`), ready to compile. |
| [`08_shared_utilities/`](08_shared_utilities/) | -- | `Shared.m` (constants: windowing, class list, evaluation protocol) and `SpectrogramDatastore.m` (custom MATLAB datastore for the spectrogram sequences), used throughout every stage above. Add this folder to your MATLAB path before running anything else. |

## How to reproduce the study end to end

1. Obtain the raw dataset (see [Data availability](#data-availability) above) and place it following
   the layout in [Directory layout you need to provide](#directory-layout-you-need-to-provide).
2. Add `08_shared_utilities/` to your MATLAB path (`addpath('08_shared_utilities')`), and run everything
   from a working directory where MATLAB can create a `Models/` subfolder for intermediate results
   (every stage below reads/writes there by timestep-stamped filename, auto-detecting the most recent
   matching file -- you never need to pass variables between scripts by hand).
3. **Stage 0 -- data preparation** (`01_data_preparation/`): run `generateSpectrogramDataset.m` (deep
   backbone path) and `generateStatFeatureDataset.m` (classical-baseline path). Run
   `addGroundTruthFromStudents.m` and `groundTruthIndexDatasetMap.m` first if your raw dataset does not
   already have `groundTruthIndex` annotations for every month (both scripts have path placeholders to
   fill in; see the comments at the top of each file).
4. **Stage 1 -- representations**:
   - `02_backbone_training/trainBackboneTransformer.m` (parametrized by capacity: `'grande'`,
     `'mediano'`, `'pequeno'`) trains the CNN-Transformer-encoder backbone; run it once per capacity.
     `trainBackboneTCN.m` trains the alternative TCN backbone. Then run `buildFeatureCache.m` to freeze
     and cache each trained backbone's representation across all seven sessions.
   - `03_classical_baselines/buildStatFeatureCache.m` builds the classical 32-d feature cache; then run
     `trainANNHead.m`, `trainSVMHead.m`, `trainKNNHead.m` (and, for the robustness sweep reported in the
     paper, `trainSVMVariants.m` / `trainKNNVariants.m`).
5. **Stage 2 -- decision heads and covariate shift** (independent of each other, can run in either
   order): `04_decision_heads/trainSoftmaxHead.m`, `context_bandit.m`, `context_bandit_adaptive.m`, then
   `compareParadigms.m` to summarize; separately, `05_covariate_shift/evaluateCovariateShift.m`.
6. **Stage 3 -- personalized models** (`06_personalized_models/`): `trainPersonalizedModels.m` (SVM +
   k-NN, one model per participant) and `trainPersonalizedSoftmax.m` (Softmax head per participant on
   the frozen small backbone).
7. **Stage 4 -- figures and paper** (`07_figures_and_paper/`): run the `generate*EN.m` scripts to
   reproduce the English-language figures used in the paper, then compile `paper/ICASSP2027-paper.tex`
   (see [Compiling the paper](#compiling-the-paper) below).

## Directory layout you need to provide

Scripts in `01_data_preparation/` expect the raw dataset under a folder structure like:

```
<your dataset root>/
  sin_video_solo_emg/
    Mes0/user4/{fist,open,pinch,waveIn,waveOut,relax}.mat
    Mes1/user4/...
    ...
    Mes6/user4/...
```

where each `.mat` file holds a `reps` struct with one cell per repetition, each containing the raw
8-channel EMG signal and (after Stage 0) a `groundTruthIndex = [start, end]` sample-index pair marking
the annotated gesture interval.

## Compiling the paper

`07_figures_and_paper/paper/` contains everything needed to compile the ICASSP-format PDF locally:
`ICASSP2027-paper.tex`, `bib.bib`, and the official ICASSP `spconf.sty` / `IEEEbib.bst` style files
(these are the standard IEEE Signal Processing Society conference template files, included here for
convenience -- if the official submission portal provides an updated author kit, prefer that one).

```bash
pdflatex -interaction=nonstopmode ICASSP2027-paper.tex
bibtex ICASSP2027-paper
pdflatex -interaction=nonstopmode ICASSP2027-paper.tex
pdflatex -interaction=nonstopmode ICASSP2027-paper.tex
```

This produces a 5-page PDF (4 pages of technical content + 1 references-only page, the ICASSP 2027
page limit) with zero LaTeX warnings when built against this repository's files.

## Hyperparameters

Every hyperparameter used in this study -- backbone training, Softmax, LinUCB, SVM, k-NN, the
covariate-shift tests, and the personalized-model configurations -- is documented in
[`docs/HYPERPARAMETERS.md`](docs/HYPERPARAMETERS.md), quoted directly from the source files with file
and line references. The paper's page limit does not leave room for this level of detail in the
prose; this table is the authoritative record. If you're trying to reproduce a specific number and the
paper's compressed description leaves something ambiguous, check here first.

## Citing

If you use this code or build on this study, please cite the paper (see `paper/ICASSP2027-paper.tex`
for the full author list and venue once published).
