% #################################################################
% Diagrama detallado de UN bloque Inception (las 4 ramas paralelas,
% con los tamanos de kernel y canales reales del codigo), para
% explicar en profundidad por que esta arquitectura tiene sentido
% para EMG. Complementa fig_pipeline_overview.png (que muestra el
% pipeline completo a alto nivel) con el detalle interno de un bloque.
% Basado en el codigo real: CNN-Transformer/trainBackboneTransformer.m,
% funcion setNeuralNetworkArchitecture, bloque Inception 1a/1f.
% #################################################################

clear; clc;

fig = figure('Visible', 'off', 'Position', [100 100 1100 720], 'Color', 'w');
ax = axes('Position', [0 0 1 1], 'Visible', 'off');
xlim(ax, [0 11]); ylim(ax, [0 7.2]); hold(ax, 'on');

colorInput = [0.85 0.85 0.85];
colorBranch = [0.70 0.85 1.00];
colorConcat = [1.00 0.85 0.60];

% --- Entrada ---
rectangle(ax, 'Position', [4.5 5.7 2 0.7], 'FaceColor', colorInput, 'EdgeColor', 'k', 'LineWidth', 1.2);
text(ax, 5.5, 6.05, 'Entrada del bloque (features de canales EMG)', 'HorizontalAlignment', 'center', 'FontSize', 9);

% --- Rama 1: 1x1 ---
rectangle(ax, 'Position', [0.3 3.6 2 0.9], 'FaceColor', colorBranch, 'EdgeColor', 'k', 'LineWidth', 1.2);
text(ax, 1.3, 4.05, sprintf('Conv 1\\times1\n(C canales)'), 'HorizontalAlignment', 'center', 'FontSize', 9);

% --- Rama 2: 3x3 via reduccion ---
rectangle(ax, 'Position', [2.6 3.3 2 1.3], 'FaceColor', colorBranch, 'EdgeColor', 'k', 'LineWidth', 1.2);
text(ax, 3.6, 3.95, sprintf('Conv 1\\times1 (R)\n\\downarrow\nConv 3\\times3 (C)'), 'HorizontalAlignment', 'center', 'FontSize', 9);

% --- Rama 3: 5x5 via reduccion ---
rectangle(ax, 'Position', [5.0 3.3 2 1.3], 'FaceColor', colorBranch, 'EdgeColor', 'k', 'LineWidth', 1.2);
text(ax, 6.0, 3.95, sprintf('Conv 1\\times1 (R)\n\\downarrow\nConv 5\\times5 (C)'), 'HorizontalAlignment', 'center', 'FontSize', 9);

% --- Rama 4: pool + proyeccion ---
rectangle(ax, 'Position', [7.4 3.3 2 1.3], 'FaceColor', colorBranch, 'EdgeColor', 'k', 'LineWidth', 1.2);
text(ax, 8.4, 3.95, sprintf('MaxPool 3\\times3\n\\downarrow\nConv 1\\times1 (C)'), 'HorizontalAlignment', 'center', 'FontSize', 9);

% --- Flechas de entrada a las 4 ramas ---
arrowTargets = [1.3, 3.6, 6.0, 8.4];
for xt = arrowTargets
    annotation(fig, 'arrow', [xt/11*0.98+0.01 xt/11*0.98+0.01], [0.785 0.66], 'LineWidth', 1.1);
end

% --- Concatenacion ---
rectangle(ax, 'Position', [3.5 1.5 4 0.9], 'FaceColor', colorConcat, 'EdgeColor', 'k', 'LineWidth', 1.5);
text(ax, 5.5, 1.95, sprintf('Concatenación por canal (depthConcatenationLayer)\n4C canales de salida'), 'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');

for xt = arrowTargets
    annotation(fig, 'arrow', [xt/11*0.98+0.01 5.5/11*0.98+0.01], [3.3/7.2 2.4/7.2], 'LineWidth', 1.1);
end

% --- Salida ---
rectangle(ax, 'Position', [4.3 0.3 2.4 0.7], 'FaceColor', colorInput, 'EdgeColor', 'k', 'LineWidth', 1.2);
text(ax, 5.5, 0.65, 'Al siguiente bloque / al Transformer', 'HorizontalAlignment', 'center', 'FontSize', 9);
annotation(fig, 'arrow', [0.50 0.50], [1.5/7.2 1.0/7.2], 'LineWidth', 1.1);

text(ax, 5.5, 6.85, 'Un bloque Inception: cuatro escalas de receptive field en paralelo', ...
    'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
text(ax, 5.5, 6.55, 'C = canales por rama, R = canales de reducción 1\times1 (R < C, controla el costo computacional)', ...
    'HorizontalAlignment', 'center', 'FontSize', 9.5);

exportgraphics(fig, 'FigurasReales/diagrama_bloque_inception.png', 'Resolution', 200);
close(fig);
fprintf('Diagrama guardado en FigurasReales/diagrama_bloque_inception.png\n');
