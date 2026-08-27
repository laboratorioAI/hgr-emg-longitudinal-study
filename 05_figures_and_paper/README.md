# Stage 4 -- Figures and Paper

Regenerates the English-language figures used in the ICASSP 2027 paper, and contains the paper's
compilable LaTeX source.

## Figure scripts

- **`generateCovariateShiftFigureEN.m`** -- reuses the already-computed statistics from
  `04_covariate_shift/evaluateCovariateShift.m`'s saved `.mat` output (no recomputation) and produces
  the English-labeled MMD²/A-distance bar-chart figure (`fig_covariate_shift_en.png`).
- **`generateUnifiedDriftFiguresEN.m`** -- reuses the Stage 1/2 result `.mat` files for the three backbone
  capacities and both decision paradigms, and produces the accuracy-by-session figure
  (`fig_sl_vs_rl_scales_en.png`, Softmax vs.\ LinUCB, three scales) reported in the paper.
- **`generateAttentionDiagram.m`**, **`generateInceptionBlockDiagram.m`**, **`generatePipelineDiagram.m`**
  -- schematic architecture diagrams (self-attention path length, Inception block, overall pipeline),
  not derived from experimental results.

Both `*EN.m` result-figure scripts are English-labeled copies of the original Spanish-labeled figure
generators used for this project's internal documentation; they read the same underlying `Models/*.mat`
result files, so re-running Stages 1--2 and then these scripts reproduces the paper's figures exactly.

## `paper/`

Contains everything needed to compile the paper locally: `ICASSP2027-paper.tex`, `bib.bib`, the ICASSP
`spconf.sty` / `IEEEbib.bst` template files, and the two English-language figures the paper includes.
See the [top-level README](../README.md#compiling-the-paper) for the compile command.
