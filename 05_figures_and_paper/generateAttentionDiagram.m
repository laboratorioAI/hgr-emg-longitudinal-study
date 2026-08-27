% Diagram comparing the maximum path length between a recurrent layer,
% a dilated convolution (TCN), and self-attention: the first-principles
% justification (Vaswani et al. 2017, Table 1) for placing self-attention
% after the CNN stage, rather than an additional recurrent or
% convolutional layer, to model temporal dependencies.

clear; clc;

fig = figure('Visible', 'off', 'Position', [100 100 1500 700], 'Color', 'w');
ax = axes('Position', [0 0 1 1], 'Visible', 'off');
xlim(ax, [0 11]); ylim(ax, [0 5]); hold(ax, 'on');

text(ax, 5.5, 4.75, 'Maximum path length between two sequence positions', ...
    'HorizontalAlignment', 'center', 'FontSize', 18, 'FontWeight', 'bold');
text(ax, 5.5, 4.45, 'Algebraic property of the mechanism, not an empirical result (Vaswani et al., 2017, Table 1)', ...
    'HorizontalAlignment', 'center', 'FontSize', 13);

n = 5; % example sequence positions
mechs = struct('name', {'Recurrent (RNN)', 'Dilated convolution (TCN)', 'Self-attention'}, ...
               'complexity', {'O(n)', 'O(log_k n)', 'O(1)'}, ...
               'x0', {0.6, 4.1, 7.6});
colorNode = [0.70 0.85 1.00];
colorEdge = [0.3 0.3 0.3];

for m = 1:3
    x0 = mechs(m).x0;
    yNodes = 2.6;
    xs = x0 + (0:n-1) * 0.55;
    for i = 1:n
        rectangle(ax, 'Position', [xs(i)-0.12 yNodes-0.12 0.24 0.24], 'Curvature', 1, ...
            'FaceColor', colorNode, 'EdgeColor', 'k', 'LineWidth', 1.2);
    end
    switch m
        case 1 % RNN: sequential chain, each step connects only to the next
            for i = 1:n-1
                annotation(fig, 'arrow', [(xs(i)+0.12)/11 (xs(i+1)-0.12)/11], [yNodes/5 yNodes/5], 'LineWidth', 1.2, 'Color', colorEdge);
            end
            for i = 1:n-1
                annotation(fig, 'arrow', [(xs(i)+0.05)/11 (xs(i+1)-0.05)/11], [(yNodes+0.4)/5 (yNodes+0.4)/5], ...
                    'LineWidth', 1.8, 'Color', [0.75 0.15 0.1], 'HeadLength', 6, 'HeadWidth', 6);
            end
        case 2 % Dilated TCN: growing skips (dilation 1,2,4)
            skips = [1 1 2 4];
            pos = 1;
            while pos < n
                step = min(skips(min(end,pos)), n-pos);
                nextPos = pos + max(step,1);
                if nextPos > n, nextPos = n; end
                annotation(fig, 'arrow', [(xs(pos)+0.05)/11 (xs(nextPos)-0.05)/11], ...
                    [(yNodes+0.4)/5 (yNodes+0.4)/5], 'LineWidth', 1.8, 'Color', [0.75 0.15 0.1]);
                pos = nextPos;
            end
        case 3 % Self-attention: every position connects directly to every other (show first-to-last)
            annotation(fig, 'arrow', [(xs(1)+0.05)/11 (xs(n)-0.05)/11], [(yNodes+0.4)/5 (yNodes+0.4)/5], ...
                'LineWidth', 2.0, 'Color', [0.1 0.55 0.2]);
            for i = 2:n-1
                plot(ax, [xs(1) xs(i)], [yNodes+0.12 yNodes+0.12], ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.0);
            end
    end
    text(ax, x0 + (n-1)*0.55/2, yNodes-0.55, mechs(m).name, 'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
    text(ax, x0 + (n-1)*0.55/2, yNodes-0.9, mechs(m).complexity, 'HorizontalAlignment', 'center', 'FontSize', 17, 'Color', [0.75 0.15 0.1], 'FontWeight', 'bold');
end

text(ax, 5.5, 0.75, ['An EMG gesture has no fixed duration or position within the window: the frame where a', ...
    newline 'contraction starts and the frame of its peak may be several steps apart. Self-attention connects', ...
    newline 'any two positions in a single hop, regardless of how far apart they are.'], ...
    'HorizontalAlignment', 'center', 'FontSize', 13);

exportgraphics(fig, 'RL_vs_SL/fig_attention_pathlength.png', 'Resolution', 220);
close(fig);
fprintf('Saved: RL_vs_SL/fig_attention_pathlength.png\n');
