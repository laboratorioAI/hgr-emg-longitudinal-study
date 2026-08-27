% #################################################################
% Diagrama comparando la longitud de camino maxima entre RNN, CNN/TCN
% dilatada, y autoatencion -- la justificacion de primeros principios
% (Vaswani et al. 2017, Tabla 1) de por que se eligio autoatencion
% despues de la CNN, en vez de una capa recurrente o convolucional
% adicional para modelar dependencias temporales.
% #################################################################

clear; clc;

fig = figure('Visible', 'off', 'Position', [100 100 1100 500], 'Color', 'w');
ax = axes('Position', [0 0 1 1], 'Visible', 'off');
xlim(ax, [0 11]); ylim(ax, [0 5]); hold(ax, 'on');

text(ax, 5.5, 4.75, 'Longitud de camino máxima entre dos posiciones de la secuencia', ...
    'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold');
text(ax, 5.5, 4.45, 'Propiedad algebraica del mecanismo, no un resultado empírico (Vaswani et al., 2017, Tabla 1)', ...
    'HorizontalAlignment', 'center', 'FontSize', 10);

n = 5; % posiciones de ejemplo en la secuencia
mechs = struct('name', {'Recurrente (RNN)', 'Convolución dilatada (TCN)', 'Autoatención'}, ...
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
            'FaceColor', colorNode, 'EdgeColor', 'k', 'LineWidth', 1);
    end
    switch m
        case 1 % RNN: cadena secuencial, cada paso conecta solo al siguiente
            for i = 1:n-1
                annotation(fig, 'arrow', [(xs(i)+0.12)/11 (xs(i+1)-0.12)/11], [yNodes/5 yNodes/5], 'LineWidth', 1, 'Color', colorEdge);
            end
            for i = 1:n-1
                annotation(fig, 'arrow', [(xs(i)+0.05)/11 (xs(i+1)-0.05)/11], [(yNodes+0.4)/5 (yNodes+0.4)/5], ...
                    'LineWidth', 1.4, 'Color', [0.75 0.15 0.1], 'HeadLength', 5, 'HeadWidth', 5);
            end
        case 2 % TCN dilatada: saltos crecientes (dilatacion 1,2,4)
            skips = [1 1 2 4];
            pos = 1;
            path = pos;
            while pos < n
                step = min(skips(min(end,pos)), n-pos);
                nextPos = pos + max(step,1);
                if nextPos > n, nextPos = n; end
                annotation(fig, 'arrow', [(xs(pos)+0.05)/11 (xs(nextPos)-0.05)/11], ...
                    [(yNodes+0.4)/5 (yNodes+0.4)/5], 'LineWidth', 1.4, 'Color', [0.75 0.15 0.1]);
                pos = nextPos;
            end
        case 3 % Autoatencion: todos conectan directo con todos (mostrar solo primero-ultimo)
            annotation(fig, 'arrow', [(xs(1)+0.05)/11 (xs(n)-0.05)/11], [(yNodes+0.4)/5 (yNodes+0.4)/5], ...
                'LineWidth', 1.6, 'Color', [0.1 0.55 0.2]);
            for i = 2:n-1
                plot(ax, [xs(1) xs(i)], [yNodes+0.12 yNodes+0.12], ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 0.8);
            end
    end
    text(ax, x0 + (n-1)*0.55/2, yNodes-0.55, mechs(m).name, 'HorizontalAlignment', 'center', 'FontSize', 10.5, 'FontWeight', 'bold');
    text(ax, x0 + (n-1)*0.55/2, yNodes-0.9, mechs(m).complexity, 'HorizontalAlignment', 'center', 'FontSize', 13, 'Color', [0.75 0.15 0.1], 'FontWeight', 'bold');
end

text(ax, 5.5, 0.75, ['Un gesto EMG no tiene duración ni posición fija dentro de la ventana: el frame de inicio de la contracción y', ...
    newline 'el de su pico pueden estar separados varios pasos. La autoatención conecta cualquier par de posiciones en un solo salto.'], ...
    'HorizontalAlignment', 'center', 'FontSize', 10);

exportgraphics(fig, 'FigurasReales/diagrama_longitud_camino.png', 'Resolution', 200);
close(fig);
fprintf('Diagrama guardado en FigurasReales/diagrama_longitud_camino.png\n');
