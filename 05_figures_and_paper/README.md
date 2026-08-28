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
The three schematic (non-data) diagrams the paper uses -- the pipeline overview
(`fig_pipeline_overview.jpg`), the Inception-block detail (`fig_inception_block.jpg`), and the
attention-vs-RNN-vs-TCN path-length comparison (`fig_attention_pathlength.jpg`) -- are **not** generated
by a script. They were originally produced with `generatePipelineDiagram.m`,
`generateInceptionBlockDiagram.m`, and `generateAttentionDiagram.m` (still present in this folder and
still functionally correct), but were replaced with versions generated externally (via a generative
image model, from prompts specifying the exact content, wording, and figure-by-figure color/line-style
legend to match) because they rendered more cleanly at print resolution. Each prompt was written to
match this repository's own code and the paper's exact claims (e.g. the Inception block's four branches
and $C$/$R$ notation come directly from `02_backbone_training/trainBackboneTransformer.m`; the
attention path-length figure's O(1)/O(n)/O(log n) claims come from Vaswani et al., Table 1, cited in the
paper). The three `.m` scripts are kept for reference and as a fallback if you'd rather regenerate them
programmatically; they are not part of the current build.

Both `*EN.m` result-figure scripts are English-labeled copies of the original Spanish-labeled figure
generators used for this project's internal documentation; they read the same underlying `Models/*.mat`
result files, so re-running Stages 1--2 and then these scripts reproduces the paper's data figures
exactly.

## `paper/`

Contains everything needed to compile the paper locally: `ICASSP2027-paper.tex`, `bib.bib`, `IEEEtran.cls`,
`IEEEbib.bst`, and the five figures the paper includes (pipeline, Inception block, and attention path
length as `.jpg`; covariate shift and the Softmax-vs-LinUCB accuracy comparison as `.png`). See the
[top-level README](../README.md#compiling-the-paper) for the compile command.
