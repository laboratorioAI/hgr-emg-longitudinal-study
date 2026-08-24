# Stage 2 (independent) -- Covariate Shift on the Frozen Representation

Tests whether the frozen 128-d (or 64-d/32-d) representation from Stage 1 itself shifts across months
-- directly, on the representation, rather than inferring it indirectly from a downstream accuracy
drop. A decision head that happens to compensate well for a real shift would still show a stable
accuracy curve even while the representation moves; this stage is designed not to be fooled by that.

## Script

**`evaluateCovariateShift.m`** -- for Month 0 vs. each of Month 1--6 independently, computes:

1. **MMD² (Maximum Mean Discrepancy)** with a Gaussian RBF kernel (median-heuristic bandwidth),
   assessed for significance with a 2000-iteration permutation test.
2. **Domain-classifier A-distance**: a 5-fold cross-validated logistic classifier tries to predict
   "Month 0 or month *m*?" from the 128-d vector; its cross-validated accuracy is converted to the
   Ben-David et al. A-distance, also assessed by permutation test.

Twelve resulting p-values (six months × two tests) are meant to be read jointly against a
Holm-Bonferroni family-wise correction, not individually -- see the paper's Results section for how
this is reported.

## Reproducibility note

The script's default scope is the large backbone's frozen representation; `nSubsample` controls how
many frames per group are used for the O(n²) kernel computation (documented in-file as a computational-
cost tradeoff, not a methodological choice). Each month uses a different but reproducible random seed
(`rng(9 + m)`) so permutation results are stable across re-runs but not identical across months by
construction.
