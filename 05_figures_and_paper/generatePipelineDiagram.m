% Pipeline block diagram for the paper's Methodology section: raw sEMG ->
% preprocessing -> spectrogram -> CNN Inception -> Transformer -> frozen
% dropout_2 -> [Softmax | LinUCB], plus the covariate-shift branch, and
% the longitudinal evaluation rule (Month 0 calibration, Month 1..6
% evaluation with no further parameter updates). Native MATLAB
% annotation-based figure, not recreated in an external tool.

clear; clc;

fig = figure('Visible', 'off', 'Position', [100 100 1500 750], 'Color', 'w');
ax = axes('Position', [0 0 1 1], 'Visible', 'off');
xlim(ax, [0 10]); ylim(ax, [0 5]); hold(ax, 'on');

colorPre = [0.85 0.85 0.85];
colorBackbone = [0.70 0.85 1.00];
colorFrozen = [1.00 0.85 0.60];
colorDecision = [0.75 0.95 0.75];
colorEval = [0.95 0.80 0.80];
fs = 13; fsSmall = 12; fsTitle = 16;

% --- Block 1: raw sEMG ---
rectangle(ax, 'Position', [0.2 3.3 1.1 1.0], 'FaceColor', colorPre, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(ax, 0.75, 3.8, sprintf('sEMG\n8 ch, 200 Hz'), 'HorizontalAlignment', 'center', 'FontSize', fs);

% --- Block 2: preprocessing + spectrogram ---
rectangle(ax, 'Position', [1.6 3.3 1.5 1.0], 'FaceColor', colorPre, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(ax, 2.35, 3.8, sprintf('Rectify +\nButterworth +\nSTFT'), 'HorizontalAlignment', 'center', 'FontSize', fsSmall);

% --- Block 3: CNN Inception ---
rectangle(ax, 'Position', [3.4 3.3 1.3 1.0], 'FaceColor', colorBackbone, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(ax, 4.05, 3.8, sprintf('CNN\nInception\n(6 blocks)'), 'HorizontalAlignment', 'center', 'FontSize', fsSmall);

% --- Block 4: Transformer ---
rectangle(ax, 'Position', [5.0 3.3 1.3 1.0], 'FaceColor', colorBackbone, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(ax, 5.65, 3.8, sprintf('Self-attention\n+ FFN\n(8 heads)'), 'HorizontalAlignment', 'center', 'FontSize', fsSmall);

% --- Block 5: frozen dropout_2 ---
rectangle(ax, 'Position', [6.6 3.3 1.3 1.0], 'FaceColor', colorFrozen, 'EdgeColor', 'k', 'LineWidth', 2);
text(ax, 7.25, 3.8, sprintf('dropout\\_2\n(frozen)\nz in R^{128}'), 'HorizontalAlignment', 'center', 'FontSize', fsSmall, 'FontWeight', 'bold');

% --- Horizontal arrows across the top row ---
arrowPairs = [1.3 1.6; 3.1 3.4; 4.7 5.0; 6.3 6.6];
for i = 1:size(arrowPairs, 1)
    annotation(fig, 'arrow', ...
        [arrowPairs(i,1)/10*0.98+0.01 arrowPairs(i,2)/10*0.98+0.01], [0.76 0.76], 'LineWidth', 1.6);
end

% --- Decision heads (below, two branches from dropout_2) ---
rectangle(ax, 'Position', [5.6 1.3 1.5 1.0], 'FaceColor', colorDecision, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(ax, 6.35, 1.8, sprintf('Softmax\n(SL)\nfrozen after\nMonth 0'), 'HorizontalAlignment', 'center', 'FontSize', fsSmall);

rectangle(ax, 'Position', [7.5 1.3 1.5 1.0], 'FaceColor', colorDecision, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(ax, 8.25, 1.8, sprintf('LinUCB\n(RL)\nfrozen after\nMonth 0'), 'HorizontalAlignment', 'center', 'FontSize', fsSmall);

% Arrows from dropout_2 to the two decision heads
annotation(fig, 'arrow', [0.73 0.66], [0.66 0.46], 'LineWidth', 1.6);
annotation(fig, 'arrow', [0.75 0.83], [0.66 0.46], 'LineWidth', 1.6);

% --- Longitudinal evaluation block (below both heads) ---
rectangle(ax, 'Position', [5.6 0.15 3.4 0.8], 'FaceColor', colorEval, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(ax, 7.3, 0.55, 'Evaluate on Month 1..6 (no further training)', 'HorizontalAlignment', 'center', 'FontSize', fsSmall);

annotation(fig, 'arrow', [0.66 0.66], [0.32 0.24], 'LineWidth', 1.6);
annotation(fig, 'arrow', [0.83 0.83], [0.32 0.24], 'LineWidth', 1.6);

% --- Covariate-shift block (separate branch from dropout_2) ---
rectangle(ax, 'Position', [0.6 1.3 3.4 1.0], 'FaceColor', colorEval, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(ax, 2.3, 1.8, sprintf('Covariate shift test: MMD^2 + permutation,\ndomain classifier + A-distance\n(Month 0 vs. each subsequent month)'), ...
    'HorizontalAlignment', 'center', 'FontSize', fsSmall);

annotation(fig, 'arrow', [0.70 0.35], [0.66 0.46], 'LineWidth', 1.6);

title(ax, 'Overall pipeline: frozen feature extractor, two decision mechanisms, covariate-shift test', ...
    'FontSize', fsTitle, 'FontWeight', 'bold', 'Position', [5 4.7 0]);

exportgraphics(fig, 'RL_vs_SL/fig_pipeline_overview.png', 'Resolution', 220);
close(fig);
fprintf('Saved: RL_vs_SL/fig_pipeline_overview.png\n');
