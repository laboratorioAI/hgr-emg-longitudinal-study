# Stage 1 (parallel) -- Classical ML Baselines

Trains the two classical, architecturally-unrelated classifiers (SVM, k-NN) that the paper uses to test
whether the drift pattern found in the deep backbone is specific to that architecture family, plus a
no-CNN dense-network baseline (ANN) used in earlier stages of the project.

## Scripts, in order

1. **`buildStatFeatureCache.m`** -- loads the 32-d hand-crafted feature dataset produced by
   `01_data_preparation/generateStatFeatureDataset.m` and assembles it into the same cache format
   (`X_train`/`y_train`/`X_val`/`y_val`/`X_test`/`y_test`/`userID_*`) that `02_backbone_training/buildFeatureCache.m`
   produces for the deep backbone -- so every downstream script can consume either cache interchangeably.

2. **`trainANNHead.m`** -- a small dense network (no convolution) on the 32-d features, used as a
   "no-CNN" control in earlier stages of this project.

3. **`trainSVMHead.m`** -- one-vs-one multiclass SVM (`fitcecoc`); selects between a linear and a
   Gaussian kernel by Month-0 validation accuracy, the same model-selection criterion used everywhere
   else in this study. Training-set size is capped at 1500 frames/class (documented in-file) for
   tractability, since multiclass SVM training scales quadratically-to-cubically with sample count.

4. **`trainKNNHead.m`** -- k-nearest-neighbors; sweeps `k` over `{1,3,5,7,9,11,15,21}` and selects by
   Month-0 validation accuracy.

5. **`trainSVMVariants.m`** / **`trainKNNVariants.m`** -- the fixed-hyperparameter robustness sweep
   reported in the paper's Discussion: additional SVM/k-NN configurations whose hyperparameters are
   *not* selected by Month-0 validation accuracy, used to test whether the magnitude of the Month-0-to-
   Month-1 accuracy step tracks model flexibility rather than being a fixed data artifact.

## Reproducibility note

All scripts here read the *most recently created* `FrozenFeatures_ann_*.mat` file in `Models/` by
timestamp. If you ever regenerate `01_data_preparation`'s output and rebuild the cache, make sure the
old cache file is deleted or that you are aware a newer one now takes precedence.
