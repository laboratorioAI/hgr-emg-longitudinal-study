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
- **`generatePipelineDiagram.m`** -- overall-pipeline schematic (`fig_pipeline_overview.png`): raw sEMG
  through preprocessing, the CNN-Inception and Transformer-encoder stages, the frozen representation,
  and both decision mechanisms, with the covariate-shift branch and the longitudinal evaluation rule.
- **`generateInceptionBlockDiagram.m`** -- detailed diagram of one Inception block
  (`fig_inception_block.png`), showing the four parallel branches with the exact kernel sizes and
  channel notation ($C$, $R$) used in `02_backbone_training/trainBackboneTransformer.m`.
- **`generateAttentionDiagram.m`** -- diagram comparing maximum path length between a recurrent layer, a
  dilated convolution, and self-attention (`fig_attention_pathlength.png`), the first-principles
  justification (Vaswani et al., Table~1) for placing self-attention after the CNN stage.

None of these three diagrams depends on experimental results (they are schematic, not data-derived), so
they only need to be regenerated if the architecture itself changes.

Both `*EN.m` result-figure scripts are English-labeled copies of the original Spanish-labeled figure
generators used for this project's internal documentation; they read the same underlying `Models/*.mat`
result files, so re-running Stages 1--2 and then these scripts reproduces the paper's data figures
exactly.

## `paper/`

Contains everything needed to compile the paper locally: `ICASSP2027-paper.tex`, `bib.bib`, `IEEEtran.cls`,
`IEEEbib.bst`, and the five figures the paper includes (pipeline, Inception block, attention path length,
covariate shift, and the Softmax-vs-LinUCB accuracy comparison). See the
[top-level README](../README.md#compiling-the-paper) for the compile command.
