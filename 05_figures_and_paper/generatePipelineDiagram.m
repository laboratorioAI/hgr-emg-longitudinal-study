% Pipeline block diagram for the paper's Methodology section: raw sEMG ->
% preprocessing -> spectrogram -> CNN Inception -> Transformer -> frozen
% dropout_2 -> [Softmax | LinUCB], plus the covariate-shift branch, and
% the longitudinal evaluation rule (Month 0 calibration, Month 1..6
% evaluation with no further parameter updates).
%
% All arrows are drawn with drawArrow() (defined at the end of this
% script, as required for a local function in a MATLAB script), which
% works directly in axes data coordinates -- the same coordinate system
% as the rectangles and text. This is deliberately NOT done with
% annotation()'s figure-normalized coordinates: those silently drift out
% of alignment whenever the figure size or axes position changes, which
% is what caused a previous version of this diagram to have visibly
% misplaced arrows.

clear; clc;

fig = figure('Visible', 'off', 'Position', [100 100 1900 1350], 'Color', 'w');
ax = axes('Position', [0.02 0.02 0.96 0.82]);
xlim(ax, [0 10.6]); ylim(ax, [0 8.7]); hold(ax, 'on');
axis(ax, 'off');

colorPre = [0.85 0.85 0.85];
colorBackbone = [0.70 0.85 1.00];
colorFrozen = [1.00 0.85 0.60];
colorDecision = [0.75 0.95 0.75];
colorEval = [0.95 0.80 0.80];
fs = 28; fsSmall = 26; fsTitle = 32;
lw = 3.2;

% --- Row layout (y-ranges) ---
% Top row (backbone pipeline):      y = [5.4, 6.5]  (2-3 line boxes)
% Middle row (decision + covshift): y = [2.5, 4.3]  (4-line boxes need more height)
% Bottom row (longitudinal eval):   y = [0.4, 1.6]

% --- Block 1: raw sEMG ---
rectangle(ax, 'Position', [0.2 5.4 1.3 1.1], 'FaceColor', colorPre, 'EdgeColor', 'k', 'LineWidth', 1.8);
text(ax, 0.85, 5.95, sprintf('sEMG\n8 ch, 200 Hz'), 'HorizontalAlignment', 'center', 'FontSize', fs);

% --- Block 2: preprocessing + spectrogram ---
rectangle(ax, 'Position', [1.9 5.4 1.7 1.1], 'FaceColor', colorPre, 'EdgeColor', 'k', 'LineWidth', 1.8);
text(ax, 2.75, 5.95, sprintf('Rectify +\nButterworth +\nSTFT'), 'HorizontalAlignment', 'center', 'FontSize', fsSmall);

% --- Block 3: CNN Inception ---
rectangle(ax, 'Position', [3.95 5.4 1.5 1.1], 'FaceColor', colorBackbone, 'EdgeColor', 'k', 'LineWidth', 1.8);
text(ax, 4.7, 5.95, sprintf('CNN\nInception\n(6 blocks)'), 'HorizontalAlignment', 'center', 'FontSize', fsSmall);

% --- Block 4: Transformer ---
rectangle(ax, 'Position', [5.8 5.4 1.5 1.1], 'FaceColor', colorBackbone, 'EdgeColor', 'k', 'LineWidth', 1.8);
text(ax, 6.55, 5.95, sprintf('Self-attention\n+ FFN\n(8 heads)'), 'HorizontalAlignment', 'center', 'FontSize', fsSmall);

% --- Block 5: frozen dropout_2 ---
rectangle(ax, 'Position', [7.65 5.4 1.6 1.1], 'FaceColor', colorFrozen, 'EdgeColor', 'k', 'LineWidth', 2.5);
text(ax, 8.45, 5.95, sprintf('dropout\\_2\n(frozen)\nz in R^{128}'), 'HorizontalAlignment', 'center', 'FontSize', fsSmall, 'FontWeight', 'bold');

% --- Horizontal arrows across the top row (data coordinates) ---
localDrawArrow(ax, 1.5, 5.95, 1.9, 5.95, lw);
localDrawArrow(ax, 3.6, 5.95, 3.95, 5.95, lw);
localDrawArrow(ax, 5.45, 5.95, 5.8, 5.95, lw);
localDrawArrow(ax, 7.3, 5.95, 7.65, 5.95, lw);

% --- Decision heads (middle row, right side, two branches from dropout_2) ---
% 4 lines of text at fsSmall need a tall box; widen too so lines like
% "frozen after" don't feel cramped against the box edges.
rectangle(ax, 'Position', [6.5 2.5 1.9 1.8], 'FaceColor', colorDecision, 'EdgeColor', 'k', 'LineWidth', 1.8);
text(ax, 7.45, 3.4, sprintf('Softmax\n(SL)\nfrozen after\nMonth 0'), 'HorizontalAlignment', 'center', 'FontSize', fsSmall);

rectangle(ax, 'Position', [8.6 2.5 1.9 1.8], 'FaceColor', colorDecision, 'EdgeColor', 'k', 'LineWidth', 1.8);
text(ax, 9.55, 3.4, sprintf('LinUCB\n(RL)\nfrozen after\nMonth 0'), 'HorizontalAlignment', 'center', 'FontSize', fsSmall);

% Arrows from dropout_2 (bottom edge, y=5.4) down to the two decision heads (top edge, y=4.3)
localDrawArrow(ax, 8.2, 5.4, 7.7, 4.4, lw);
localDrawArrow(ax, 8.7, 5.4, 9.4, 4.4, lw);

% --- Longitudinal evaluation block (bottom row, spans below both heads) ---
rectangle(ax, 'Position', [6.5 0.4 4.0 1.2], 'FaceColor', colorEval, 'EdgeColor', 'k', 'LineWidth', 1.8);
text(ax, 8.5, 1.0, sprintf('Evaluate on Month 1..6\n(no further training)'), 'HorizontalAlignment', 'center', 'FontSize', fsSmall);

localDrawArrow(ax, 7.45, 2.5, 7.45, 1.6, lw);
localDrawArrow(ax, 9.55, 2.5, 9.55, 1.6, lw);

% --- Covariate-shift block (middle row, left side, separate branch from dropout_2) ---
rectangle(ax, 'Position', [0.2 2.7 5.4 1.4], 'FaceColor', colorEval, 'EdgeColor', 'k', 'LineWidth', 1.8);
text(ax, 2.9, 3.4, sprintf('Covariate shift test: MMD^2 + permutation,\ndomain classifier + A-distance\n(Month 0 vs. each subsequent month)'), ...
    'HorizontalAlignment', 'center', 'FontSize', fsSmall);

% Arrow from dropout_2 (bottom-left corner area) diagonally down-left to the covariate-shift block
localDrawArrow(ax, 7.65, 5.4, 5.7, 4.1, lw);

text(ax, 5.3, 8.1, sprintf('Overall pipeline: frozen feature extractor,\ntwo decision mechanisms, covariate-shift test'), ...
    'HorizontalAlignment', 'center', 'FontSize', fsTitle, 'FontWeight', 'bold');

exportgraphics(fig, 'RL_vs_SL/fig_pipeline_overview.png', 'Resolution', 220);
close(fig);
fprintf('Saved: RL_vs_SL/fig_pipeline_overview.png\n');

function localDrawArrow(ax, xFrom, yFrom, xTo, yTo, lw)
    dx = xTo - xFrom; dy = yTo - yFrom;
    quiver(ax, xFrom, yFrom, dx, dy, 0, 'Color', 'k', 'LineWidth', lw, ...
        'MaxHeadSize', 1.0, 'AutoScale', 'off');
end
