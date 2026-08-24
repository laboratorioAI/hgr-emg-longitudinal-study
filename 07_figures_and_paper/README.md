# Stage 4 -- Figures and Paper

Regenerates the English-language figures used in the ICASSP 2027 paper, and contains the paper's
compilable LaTeX source.

## Figure scripts

- **`generateCovariateShiftFigureEN.m`** -- reuses the already-computed statistics from
  `05_covariate_shift/evaluateCovariateShift.m`'s saved `.mat` output (no recomputation) and produces
  the English-labeled MMD²/A-distance bar-chart figure (`fig_covariate_shift_en.png`).
- **`generateUnifiedDriftFiguresEN.m`** -- reuses the Stage 1/2 result `.mat` files for all five models
  (three backbone capacities × Softmax/LinUCB, plus SVM/k-NN) and produces the two-panel
  accuracy-by-session figure (`fig_sl_vs_rl_scales_en.png`).
- **`generatePersonalizedFiguresEN.m`** -- reuses the Stage 3 result `.mat` files and produces the
  between-participant-dispersion-by-session figure (`fig_personalizado_sd_por_mes_en.png`).
- **`generateAttentionDiagram.m`**, **`generateInceptionBlockDiagram.m`**, **`generatePipelineDiagram.m`**
  -- schematic architecture diagrams (self-attention path length, Inception block, overall pipeline),
  not derived from experimental results.

All three `*EN.m` scripts are English-labeled copies of the original Spanish-labeled figure generators
used for this project's internal documentation; they read the same underlying `Models/*.mat` result
files, so re-running Stages 1--3 and then these scripts reproduces the paper's figures exactly.

## `paper/`

Contains everything needed to compile the paper locally: `ICASSP2027-paper.tex`, `bib.bib`, the ICASSP
`spconf.sty` / `IEEEbib.bst` template files, and the three English-language figures the paper includes.
See the [top-level README](../README.md#compiling-the-paper) for the compile command.
