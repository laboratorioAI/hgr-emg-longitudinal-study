% Diagram comparing the maximum path length between a recurrent layer, a
% dilated convolution (TCN), and self-attention: the first-principles
% justification (Vaswani et al. 2017, Table 1) for placing self-attention
% after the CNN stage, rather than an additional recurrent or
% convolutional layer, to model temporal dependencies.
%
% Each of the three mechanisms gets its own row (not sharing a row with
% the others), with sequence positions numbered 1..5 and a single
% highlighted worst-case path from position 1 to position 5 drawn in a
% distinct color, plus the full connectivity for the sequential/
% recurrent case. This replaces an earlier, more cluttered version of
% this figure. Arrows/lines use quiver()/plot() in axes data
% coordinates, not annotation()'s figure-normalized coordinates.

clear; clc;

fig = figure('Visible', 'off', 'Position', [100 100 1700 1050], 'Color', 'w');
ax = axes('Position', [0.04 0.04 0.92 0.86]);
xlim(ax, [0 11]); ylim(ax, [0 10]); hold(ax, 'on');
axis(ax, 'off');

text(ax, 5.5, 9.6, 'Maximum path length between two sequence positions', ...
    'HorizontalAlignment', 'center', 'FontSize', 22, 'FontWeight', 'bold');
text(ax, 5.5, 9.05, 'Algebraic property of the mechanism, not an empirical result (Vaswani et al., 2017, Table 1)', ...
    'HorizontalAlignment', 'center', 'FontSize', 15);

n = 5; % example sequence positions
xNodes = 2.0 + (0:n-1) * 1.4; % shared x-positions for all three rows
rNadius = 0.22;
colorNode = [0.70 0.85 1.00];
colorSeqEdge = [0.55 0.55 0.55];
colorHighlight = [0.80 0.10 0.10];
colorGood = [0.10 0.55 0.20];

rowY = [7.0, 4.4, 1.8]; % RNN, TCN, self-attention rows (top to bottom)
rowNames = {'Recurrent (RNN)', 'Dilated convolution (TCN)', 'Self-attention'};
rowComplexity = {'O(n)', 'O(log_k n)', 'O(1)'};

% Row separators for visual clarity
for r = 1:2
    ySep = (rowY(r) + rowY(r+1)) / 2 + 0.3;
    plot(ax, [0.5 10.5], [ySep ySep], ':', 'Color', [0.8 0.8 0.8], 'LineWidth', 1);
end

% ============================== ROW 1: RNN ==============================
y = rowY(1);
for i = 1:n
    rectangle(ax, 'Position', [xNodes(i)-rNadius y-rNadius 2*rNadius 2*rNadius], 'Curvature', 1, ...
        'FaceColor', colorNode, 'EdgeColor', 'k', 'LineWidth', 1.5);
    text(ax, xNodes(i), y, num2str(i), 'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold');
end
% sequential hops (short gray arrows, one per adjacent pair)
for i = 1:n-1
    localDrawArrow(ax, xNodes(i)+rNadius, y, xNodes(i+1)-rNadius, y, 1.6, colorSeqEdge, 0.5);
end
% the path from node 1 to node 5 requires traversing every hop: show as
% a curved highlighted path above the chain, labeled with the hop count
yArc = y + 0.85;
plot(ax, [xNodes(1) xNodes(1) xNodes(end) xNodes(end)], [y+rNadius yArc yArc y+rNadius], '-', ...
    'Color', colorHighlight, 'LineWidth', 2.4);
for i = 1:n-1
    xm = (xNodes(i)+xNodes(i+1))/2;
    text(ax, xm, yArc+0.28, num2str(i), 'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', colorHighlight);
end
text(ax, xNodes(end)+1.1, y, sprintf('%s\n%s', rowNames{1}, rowComplexity{1}), ...
    'HorizontalAlignment', 'left', 'FontSize', 17, 'FontWeight', 'bold', 'Color', colorHighlight);
text(ax, 0.9, y, '4 sequential hops from position 1 to position 5', ...
    'HorizontalAlignment', 'left', 'FontSize', 12.5, 'Color', [0.3 0.3 0.3], 'Position', [0.3 y-0.75 0]);

% ============================== ROW 2: TCN ==============================
y = rowY(2);
for i = 1:n
    rectangle(ax, 'Position', [xNodes(i)-rNadius y-rNadius 2*rNadius 2*rNadius], 'Curvature', 1, ...
        'FaceColor', colorNode, 'EdgeColor', 'k', 'LineWidth', 1.5);
    text(ax, xNodes(i), y, num2str(i), 'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold');
end
% dilated skips: 1->2 (dilation 1), 2->3 (dilation 1), 3->5 (dilation 2)
% i.e. only 3 hops instead of 4, growing exponentially with layer depth
skipPairs = [1 2; 2 3; 3 5];
for k = 1:size(skipPairs,1)
    i = skipPairs(k,1); j = skipPairs(k,2);
    if j - i == 1
        localDrawArrow(ax, xNodes(i)+rNadius, y, xNodes(j)-rNadius, y, 1.6, colorSeqEdge, 0.5);
    else
        yArcLocal = y + 0.55;
        plot(ax, [xNodes(i) xNodes(i) xNodes(j) xNodes(j)], [y+rNadius yArcLocal yArcLocal y+rNadius], '-', ...
            'Color', colorSeqEdge, 'LineWidth', 1.6);
    end
end
yArc = y + 0.95;
plot(ax, [xNodes(1) xNodes(1) xNodes(end) xNodes(end)], [y+rNadius yArc yArc y+rNadius], '-', ...
    'Color', colorHighlight, 'LineWidth', 2.4);
text(ax, (xNodes(1)+xNodes(end))/2, yArc+0.28, '3 hops (dilation 1, 1, 2)', ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', colorHighlight);
text(ax, xNodes(end)+1.1, y, sprintf('%s\n%s', rowNames{2}, rowComplexity{2}), ...
    'HorizontalAlignment', 'left', 'FontSize', 17, 'FontWeight', 'bold', 'Color', colorHighlight);

% ============================== ROW 3: Self-attention ==============================
y = rowY(3);
for i = 1:n
    rectangle(ax, 'Position', [xNodes(i)-rNadius y-rNadius 2*rNadius 2*rNadius], 'Curvature', 1, ...
        'FaceColor', colorNode, 'EdgeColor', 'k', 'LineWidth', 1.5);
    text(ax, xNodes(i), y, num2str(i), 'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold');
end
% every position connects to every other in one hop: draw all pairwise
% connections as light dotted lines, then highlight the 1-to-5 hop
for i = 1:n
    for j = i+1:n
        plot(ax, [xNodes(i) xNodes(j)], [y y], ':', 'Color', [0.75 0.75 0.75], 'LineWidth', 1);
    end
end
yArc = y + 0.55;
plot(ax, [xNodes(1) xNodes(1) xNodes(end) xNodes(end)], [y+rNadius yArc yArc y+rNadius], '-', ...
    'Color', colorGood, 'LineWidth', 2.6);
text(ax, (xNodes(1)+xNodes(end))/2, yArc+0.28, '1 hop, any pair', ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', colorGood, 'FontWeight', 'bold');
text(ax, xNodes(end)+1.1, y, sprintf('%s\n%s', rowNames{3}, rowComplexity{3}), ...
    'HorizontalAlignment', 'left', 'FontSize', 17, 'FontWeight', 'bold', 'Color', colorGood);

text(ax, 5.5, 0.55, ['An EMG gesture has no fixed duration or position within the window: the frame where a contraction starts and', ...
    newline 'the frame of its peak may be several steps apart. Self-attention connects any two positions in a single hop.'], ...
    'HorizontalAlignment', 'center', 'FontSize', 13.5);

exportgraphics(fig, 'RL_vs_SL/fig_attention_pathlength.png', 'Resolution', 220);
close(fig);
fprintf('Saved: RL_vs_SL/fig_attention_pathlength.png\n');

function localDrawArrow(ax, xFrom, yFrom, xTo, yTo, lw, color, headSize)
    dx = xTo - xFrom; dy = yTo - yFrom;
    quiver(ax, xFrom, yFrom, dx, dy, 0, 'Color', color, 'LineWidth', lw, ...
        'MaxHeadSize', headSize, 'AutoScale', 'off');
end
