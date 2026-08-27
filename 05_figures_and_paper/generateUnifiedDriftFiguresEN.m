% #################################################################
% Regenera la figura central del paper (fig_sl_vs_rl_scales_en.png):
% accuracy por sesion, Softmax (SL) vs. LinUCB (RL), en las 3 capacidades
% del backbone CNN-Transformer-encoder (grande/mediano/pequeno). El
% paper reporta solo esta comparacion (backbone + 2 paradigmas de
% decision); no incluye SVM/kNN ni modelos personalizados.
% #################################################################

clear; clc;

variantes = {'grande', 'mediano', 'pequeno'};
etiquetasVariante = {'Grande (3.03M)', 'Mediano (1.00M)', 'Pequeño (0.25M)'};
numParams = [3026614, 998774, 246850];

resultados = struct();
for v = 1:numel(variantes)
    tam = variantes{v};
    if strcmp(tam, 'grande')
        fs = dir(fullfile('Models', 'SoftmaxHead_Results_*.mat'));
        fs = fs(~contains({fs.name}, {'_ann_', '_mediano_', '_pequeno_', '_tcn_', '_svm_', '_knn_', '_micro_', '_nano_'}));
        fb = dir(fullfile('Models', 'Bandit_Results_*.mat'));
        fb = fb(~contains({fb.name}, {'_ann_', '_mediano_', '_pequeno_', '_tcn_', '_svm_', '_knn_', '_micro_', '_nano_'}));
    else
        fs = dir(fullfile('Models', sprintf('SoftmaxHead_Results_%s_*.mat', tam)));
        fb = dir(fullfile('Models', sprintf('Bandit_Results_%s_*.mat', tam)));
    end
    [~, idxS] = max([fs.datenum]);
    [~, idxB] = max([fb.datenum]);
    ls = load(fullfile(fs(idxS).folder, fs(idxS).name), 'accByMonth', 'monthLabels');
    lb = load(fullfile(fb(idxB).folder, fb(idxB).name), 'accByMonth');
    resultados.(tam).softmax = ls.accByMonth * 100;
    resultados.(tam).bandit = lb.accByMonth * 100;
    resultados.(tam).monthLabels = cellstr(ls.monthLabels);
end

monthLabels = resultados.grande.monthLabels;
xVals = 1:numel(monthLabels);

outDir = fullfile('RL_vs_SL');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

colorSoftmax = [0.20 0.35 0.60];
colorBandit = [0.85 0.33 0.10];
colorSvm = [0.30 0.60 0.30];
colorKnn = [0.55 0.35 0.65];
estiloEscala = {'-o', '--s', ':^'};
grosorEscala = [2.5, 2.0, 1.5];

%% ======================================================================
%% FIGURA 1: fig_sl_vs_rl_scales.png -- Softmax (SL) vs. LinUCB (RL), 3 escalas
%% ======================================================================
fig = figure('Visible', 'off', 'Position', [850 100 950 850]);

hold on;
for v = 1:numel(variantes)
    tam = variantes{v};
    plot(xVals, resultados.(tam).softmax, estiloEscala{v}, 'Color', colorSoftmax, ...
        'LineWidth', grosorEscala(v)+1, 'MarkerFaceColor', colorSoftmax, 'MarkerSize', 9, ...
        'DisplayName', sprintf('Softmax (SL) - %s', etiquetasVariante{v}));
end
for v = 1:numel(variantes)
    tam = variantes{v};
    plot(xVals, resultados.(tam).bandit, estiloEscala{v}, 'Color', colorBandit, ...
        'LineWidth', grosorEscala(v)+1, 'MarkerFaceColor', colorBandit, 'MarkerSize', 9, ...
        'DisplayName', sprintf('Frozen LinUCB (RL) - %s', etiquetasVariante{v}));
end
hold off;
grid on;
xticks(xVals); xticklabels(monthLabels);
ylabel('Accuracy (%)'); xlabel('Session');
title({'CNN-Transformer-encoder backbone:', 'Softmax (SL) vs. frozen LinUCB (RL), 3 scales'});
legend('Location', 'southoutside', 'NumColumns', 2, 'FontSize', 13);
ylim([70 100]);
set(gca, 'FontSize', 17);
set(get(gca,'XLabel'), 'FontSize', 18);
set(get(gca,'YLabel'), 'FontSize', 18);
set(get(gca,'Title'), 'FontSize', 18, 'FontWeight', 'bold');

exportgraphics(fig, fullfile(outDir, 'fig_sl_vs_rl_scales_en.png'), 'Resolution', 200);
close(fig);
fprintf('Saved: %s\n', fullfile(outDir, 'fig_sl_vs_rl_scales_en.png'));
return; % ICASSP-paper variant only needs Figure 1 (Softmax vs LinUCB, 3 scales)
