% Detailed diagram of one Inception block (the four parallel branches,
% with the actual kernel sizes and channel counts from the code), so a
% reader can reproduce the exact architecture. Complements
% fig_pipeline_overview.png (which shows the overall pipeline at a high
% level) with the internal detail of a single block. Based on the real
% code: 02_backbone_training/trainBackboneTransformer.m, function
% setNeuralNetworkArchitecture, Inception block 1a/1f.
%
% Arrows are drawn with localDrawArrow() (defined at the end of this
% script) directly in axes data coordinates, not with annotation()'s
% figure-normalized coordinates, which drift out of alignment whenever
% figure size or axes position changes.

clear; clc;

fig = figure('Visible', 'off', 'Position', [100 100 2100 1500], 'Color', 'w');
ax = axes('Position', [0.02 0.02 0.96 0.90]);
xlim(ax, [0 11]); ylim(ax, [0 7.6]); hold(ax, 'on');
axis(ax, 'off');

colorInput = [0.85 0.85 0.85];
colorBranch = [0.70 0.85 1.00];
colorConcat = [1.00 0.85 0.60];
fs = 26; fsSmall = 24;
lw = 3.2;

% --- Input ---
rectangle(ax, 'Position', [3.8 5.6 3.4 1.0], 'FaceColor', colorInput, 'EdgeColor', 'k', 'LineWidth', 2);
text(ax, 5.5, 6.1, sprintf('Block input\n(EMG-channel features)'), 'HorizontalAlignment', 'center', 'FontSize', fs);

% --- Branch 1: 1x1 ---
rectangle(ax, 'Position', [0.2 3.55 2.2 1.15], 'FaceColor', colorBranch, 'EdgeColor', 'k', 'LineWidth', 2);
text(ax, 1.3, 4.125, sprintf('Conv 1\\times1\n(C channels)'), 'HorizontalAlignment', 'center', 'FontSize', fs);

% --- Branch 2: 3x3 via reduction ---
rectangle(ax, 'Position', [2.65 3.15 2.2 1.55], 'FaceColor', colorBranch, 'EdgeColor', 'k', 'LineWidth', 2);
text(ax, 3.75, 3.925, sprintf('Conv 1\\times1 (R)\n\\downarrow\nConv 3\\times3 (C)'), 'HorizontalAlignment', 'center', 'FontSize', fs);

% --- Branch 3: 5x5 via reduction ---
rectangle(ax, 'Position', [5.15 3.15 2.2 1.55], 'FaceColor', colorBranch, 'EdgeColor', 'k', 'LineWidth', 2);
text(ax, 6.25, 3.925, sprintf('Conv 1\\times1 (R)\n\\downarrow\nConv 5\\times5 (C)'), 'HorizontalAlignment', 'center', 'FontSize', fs);

% --- Branch 4: pool + projection ---
rectangle(ax, 'Position', [7.65 3.15 2.2 1.55], 'FaceColor', colorBranch, 'EdgeColor', 'k', 'LineWidth', 2);
text(ax, 8.75, 3.925, sprintf('MaxPool 3\\times3\n\\downarrow\nConv 1\\times1 (C)'), 'HorizontalAlignment', 'center', 'FontSize', fs);

% --- Arrows from input (bottom edge, y=5.6) to each branch's top edge ---
branchTopY = [4.7, 4.7, 4.7, 4.7]; % top edge of each branch box (branch 1 is shorter)
branchX = [1.3, 3.75, 6.25, 8.75];
for i = 1:4
    localDrawArrow(ax, branchX(i), 5.6, branchX(i), branchTopY(i), lw);
end

% --- Concatenation ---
rectangle(ax, 'Position', [2.9 1.5 5.2 1.15], 'FaceColor', colorConcat, 'EdgeColor', 'k', 'LineWidth', 2.6);
text(ax, 5.5, 2.075, sprintf('Channel-wise concatenation\n(depthConcatenationLayer), 4C output channels'), 'HorizontalAlignment', 'center', 'FontSize', fs, 'FontWeight', 'bold');

% --- Arrows from each branch's bottom edge down to the concat block's top edge ---
% Each arrow lands at a distinct point along the concat block's top edge
% (spread out under its own branch), rather than all four converging on
% the exact same pixel, which reads as a tangled knot at this scale.
branchBottomY = [3.55, 3.15, 3.15, 3.15];
concatTargetX = [3.7, 4.5, 6.5, 7.3];
for i = 1:4
    localDrawArrow(ax, branchX(i), branchBottomY(i), concatTargetX(i), 2.65, lw);
end

% --- Output ---
rectangle(ax, 'Position', [3.6 0.25 3.8 0.85], 'FaceColor', colorInput, 'EdgeColor', 'k', 'LineWidth', 2);
text(ax, 5.5, 0.675, 'To the next block / to the Transformer', 'HorizontalAlignment', 'center', 'FontSize', fs);
localDrawArrow(ax, 5.5, 1.5, 5.5, 1.1, lw);

text(ax, 5.5, 7.35, 'One Inception block: four receptive-field scales in parallel', ...
    'HorizontalAlignment', 'center', 'FontSize', 32, 'FontWeight', 'bold');
text(ax, 5.5, 6.85, 'C = channels per branch, R = 1\times1 reduction channels (R < C, controls computational cost)', ...
    'HorizontalAlignment', 'center', 'FontSize', fsSmall);

exportgraphics(fig, 'RL_vs_SL/fig_inception_block.png', 'Resolution', 220);
close(fig);
fprintf('Saved: RL_vs_SL/fig_inception_block.png\n');

function localDrawArrow(ax, xFrom, yFrom, xTo, yTo, lw)
    dx = xTo - xFrom; dy = yTo - yFrom;
    quiver(ax, xFrom, yFrom, dx, dy, 0, 'Color', 'k', 'LineWidth', lw, ...
        'MaxHeadSize', 0.6, 'AutoScale', 'off');
end
