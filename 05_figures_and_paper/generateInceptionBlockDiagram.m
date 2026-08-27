% Detailed diagram of one Inception block (the four parallel branches,
% with the actual kernel sizes and channel counts from the code), so a
% reader can reproduce the exact architecture. Complements
% fig_pipeline_overview.png (which shows the overall pipeline at a
% high level) with the internal detail of a single block. Based on the
% real code: 02_backbone_training/trainBackboneTransformer.m, function
% setNeuralNetworkArchitecture, Inception block 1a/1f.

clear; clc;

fig = figure('Visible', 'off', 'Position', [100 100 1500 950], 'Color', 'w');
ax = axes('Position', [0 0 1 1], 'Visible', 'off');
xlim(ax, [0 11]); ylim(ax, [0 7.2]); hold(ax, 'on');

colorInput = [0.85 0.85 0.85];
colorBranch = [0.70 0.85 1.00];
colorConcat = [1.00 0.85 0.60];
fs = 14; fsSmall = 13;

% --- Input ---
rectangle(ax, 'Position', [4.0 5.65 3 0.8], 'FaceColor', colorInput, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(ax, 5.5, 6.05, sprintf('Block input\n(EMG-channel features)'), 'HorizontalAlignment', 'center', 'FontSize', fs);

% --- Branch 1: 1x1 ---
rectangle(ax, 'Position', [0.3 3.6 2 0.9], 'FaceColor', colorBranch, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(ax, 1.3, 4.05, sprintf('Conv 1\\times1\n(C channels)'), 'HorizontalAlignment', 'center', 'FontSize', fs);

% --- Branch 2: 3x3 via reduction ---
rectangle(ax, 'Position', [2.6 3.3 2 1.3], 'FaceColor', colorBranch, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(ax, 3.6, 3.95, sprintf('Conv 1\\times1 (R)\n\\downarrow\nConv 3\\times3 (C)'), 'HorizontalAlignment', 'center', 'FontSize', fs);

% --- Branch 3: 5x5 via reduction ---
rectangle(ax, 'Position', [5.0 3.3 2 1.3], 'FaceColor', colorBranch, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(ax, 6.0, 3.95, sprintf('Conv 1\\times1 (R)\n\\downarrow\nConv 5\\times5 (C)'), 'HorizontalAlignment', 'center', 'FontSize', fs);

% --- Branch 4: pool + projection ---
rectangle(ax, 'Position', [7.4 3.3 2 1.3], 'FaceColor', colorBranch, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(ax, 8.4, 3.95, sprintf('MaxPool 3\\times3\n\\downarrow\nConv 1\\times1 (C)'), 'HorizontalAlignment', 'center', 'FontSize', fs);

% --- Arrows from input to the four branches ---
arrowTargets = [1.3, 3.6, 6.0, 8.4];
for xt = arrowTargets
    annotation(fig, 'arrow', [xt/11*0.98+0.01 xt/11*0.98+0.01], [0.785 0.66], 'LineWidth', 1.4);
end

% --- Concatenation ---
rectangle(ax, 'Position', [3.5 1.5 4 0.9], 'FaceColor', colorConcat, 'EdgeColor', 'k', 'LineWidth', 2);
text(ax, 5.5, 1.95, sprintf('Channel-wise concatenation (depthConcatenationLayer)\n4C output channels'), 'HorizontalAlignment', 'center', 'FontSize', fs, 'FontWeight', 'bold');

for xt = arrowTargets
    annotation(fig, 'arrow', [xt/11*0.98+0.01 5.5/11*0.98+0.01], [3.3/7.2 2.4/7.2], 'LineWidth', 1.4);
end

% --- Output ---
rectangle(ax, 'Position', [4.3 0.3 2.4 0.7], 'FaceColor', colorInput, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(ax, 5.5, 0.65, 'To the next block / to the Transformer', 'HorizontalAlignment', 'center', 'FontSize', fs);
annotation(fig, 'arrow', [0.50 0.50], [1.5/7.2 1.0/7.2], 'LineWidth', 1.4);

text(ax, 5.5, 6.85, 'One Inception block: four receptive-field scales in parallel', ...
    'HorizontalAlignment', 'center', 'FontSize', 17, 'FontWeight', 'bold');
text(ax, 5.5, 6.55, 'C = channels per branch, R = 1\times1 reduction channels (R < C, controls computational cost)', ...
    'HorizontalAlignment', 'center', 'FontSize', fsSmall);

exportgraphics(fig, 'RL_vs_SL/fig_inception_block.png', 'Resolution', 220);
close(fig);
fprintf('Saved: RL_vs_SL/fig_inception_block.png\n');
