% #################################################################
% Regenera las 4 figuras centrales del paper (fig_sl_vs_rl_scales,
% fig_drift_barras, fig_drift_caida_relativa, fig_drift_vs_capacidad)
% integrando SVM y kNN junto a los 3 backbones CNN-Transformer-encoder
% (grande/mediano/pequeno), en vez de dejarlos en una tabla aislada.
% Pedido explicito del usuario (2026-08-06): "la idea es trabajar con
% todas estas variantes juntas... las comparaciones deben ser
% consistentes" -- no dos investigaciones separadas.
%
% SVM y kNN no tienen una variante LinUCB (no hay paradigma dual para
% ellos), asi que en la figura de lineas por mes van en un panel B aparte
% (misma figura, ejes propios porque su rango de accuracy es muy distinto
% del rango 70-100% de los backbones Transformer). En las otras 3 figuras
% (barras M0-vs-M6, caida relativa, caida-vs-capacidad) se agregan como
% 2 series/categorias mas junto a las 3 escalas de backbone.
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

% --- SVM y kNN: solo un clasificador cada uno (sin paradigma dual) ---
% Usar los .mat con split corregido (29-07-2026 en adelante).
fSvm = dir(fullfile('Models', 'SoftmaxHead_Results_svm_*.mat'));
fSvm = fSvm([fSvm.datenum] > datenum('28-07-2026', 'dd-mm-yyyy'));
[~, idxSvm] = max([fSvm.datenum]);
lSvm = load(fullfile(fSvm(idxSvm).folder, fSvm(idxSvm).name), 'accByMonth', 'monthLabels');

fKnn = dir(fullfile('Models', 'SoftmaxHead_Results_knn_*.mat'));
fKnn = fKnn([fKnn.datenum] > datenum('28-07-2026', 'dd-mm-yyyy'));
[~, idxKnn] = max([fKnn.datenum]);
lKnn = load(fullfile(fKnn(idxKnn).folder, fKnn(idxKnn).name), 'accByMonth', 'monthLabels');

resultados.svm.acc = lSvm.accByMonth * 100;
resultados.knn.acc = lKnn.accByMonth * 100;
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
%% FIGURA 1: fig_sl_vs_rl_scales.png -- dos paneles (SL/RL 3 escalas | SVM+kNN)
%% ======================================================================
fig = figure('Visible', 'off', 'Position', [100 100 1500 650]);

subplot(1,2,1);
hold on;
for v = 1:numel(variantes)
    tam = variantes{v};
    plot(xVals, resultados.(tam).softmax, estiloEscala{v}, 'Color', colorSoftmax, ...
        'LineWidth', grosorEscala(v), 'MarkerFaceColor', colorSoftmax, 'MarkerSize', 6, ...
        'DisplayName', sprintf('Softmax (SL) - %s', etiquetasVariante{v}));
end
for v = 1:numel(variantes)
    tam = variantes{v};
    plot(xVals, resultados.(tam).bandit, estiloEscala{v}, 'Color', colorBandit, ...
        'LineWidth', grosorEscala(v), 'MarkerFaceColor', colorBandit, 'MarkerSize', 6, ...
        'DisplayName', sprintf('Frozen LinUCB (RL) - %s', etiquetasVariante{v}));
end
hold off;
grid on;
xticks(xVals); xticklabels(monthLabels);
ylabel('Accuracy (%)'); xlabel('Session');
title({'Panel A: CNN-Transformer-encoder', 'Softmax (SL) vs. frozen LinUCB (RL), 3 scales'});
legend('Location', 'southoutside', 'NumColumns', 2, 'FontSize', 8);
ylim([70 100]);

subplot(1,2,2);
hold on;
plot(xVals, resultados.svm.acc, '-d', 'Color', colorSvm, 'LineWidth', 2.2, ...
    'MarkerFaceColor', colorSvm, 'MarkerSize', 7, 'DisplayName', 'SVM (Gaussian kernel)');
plot(xVals, resultados.knn.acc, '-p', 'Color', colorKnn, 'LineWidth', 2.2, ...
    'MarkerFaceColor', colorKnn, 'MarkerSize', 8, 'DisplayName', 'k-NN (k=1)');
hold off;
grid on;
xticks(xVals); xticklabels(monthLabels);
ylabel('Accuracy (%)'); xlabel('Session');
title({'Panel B: classical classifiers', 'on 32-d statistical EMG features'});
legend('Location', 'southoutside', 'NumColumns', 1, 'FontSize', 9);
ylim([40 100]);

sgtitle('Accuracy by session: CNN-Transformer-encoder backbone (A) and classical classifiers (B)', ...
    'FontWeight', 'bold', 'FontSize', 13);

exportgraphics(fig, fullfile(outDir, 'fig_sl_vs_rl_scales_en.png'), 'Resolution', 200);
close(fig);
fprintf('Saved: %s\n', fullfile(outDir, 'fig_sl_vs_rl_scales_en.png'));
return; % ICASSP-paper variant only needs Figure 1 (Panel A/B accuracy-by-session)

%% ======================================================================
%% FIGURA 2: fig_drift_barras.png -- ahora con 5 grupos de barras
%% ======================================================================
gruposEtiquetas = {'Grande', 'Mediano', 'Pequeño', 'SVM', 'k-NN'};
m0vals = [resultados.grande.softmax(1), resultados.mediano.softmax(1), resultados.pequeno.softmax(1), ...
          resultados.svm.acc(1), resultados.knn.acc(1)];
m6vals = [resultados.grande.softmax(end), resultados.mediano.softmax(end), resultados.pequeno.softmax(end), ...
          resultados.svm.acc(end), resultados.knn.acc(end)];
m0valsRL = [resultados.grande.bandit(1), resultados.mediano.bandit(1), resultados.pequeno.bandit(1), NaN, NaN];
m6valsRL = [resultados.grande.bandit(end), resultados.mediano.bandit(end), resultados.pequeno.bandit(end), NaN, NaN];

fig = figure('Visible', 'off', 'Position', [100 100 1400 650]);
hold on;
nGrupos = 5;
barWidth = 0.19;
xBase = 1:nGrupos;
offsets = [-1.5 -0.5 0.5 1.5] * barWidth;

b1 = bar(xBase + offsets(1), m0vals, barWidth, 'FaceColor', colorSoftmax, 'DisplayName', 'SL / clásico — Mes0(val)');
b2 = bar(xBase + offsets(2), m6vals, barWidth, 'FaceColor', [0.65 0.75 0.88], 'DisplayName', 'SL / clásico — Mes6');
b3 = bar(xBase + offsets(3), m0valsRL, barWidth, 'FaceColor', colorBandit, 'DisplayName', 'RL (LinUCB) — Mes0(val)');
b4 = bar(xBase + offsets(4), m6valsRL, barWidth, 'FaceColor', [0.95 0.72 0.55], 'DisplayName', 'RL (LinUCB) — Mes6');
hold off;
grid on;
xticks(xBase); xticklabels(gruposEtiquetas);
ylabel('Accuracy (%)');
title('Accuracy en Mes0(val) vs. Mes6, 5 modelos: 3 backbones (SL+RL) y 2 clasificadores clásicos');
legend('Location', 'southoutside', 'NumColumns', 2, 'FontSize', 9);
ylim([0 100]);
xline(3.5, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'HandleVisibility', 'off');
text(2, 96, 'CNN-Transformer-encoder (SL + RL)', 'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', [0.4 0.4 0.4]);
text(4.5, 96, 'Clásicos (features 32-d)', 'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', [0.4 0.4 0.4]);

exportgraphics(fig, fullfile(outDir, 'fig_drift_barras.png'), 'Resolution', 200);
close(fig);
fprintf('Guardado: %s\n', fullfile(outDir, 'fig_drift_barras.png'));

%% ======================================================================
%% FIGURA 3: fig_drift_caida_relativa.png -- 5 modelos (SL curvas + RL curvas + SVM + kNN)
%% ======================================================================
fig = figure('Visible', 'off', 'Position', [100 100 1100 750]);
hold on;
for v = 1:numel(variantes)
    tam = variantes{v};
    caidaSL = (resultados.(tam).softmax(1) - resultados.(tam).softmax) / resultados.(tam).softmax(1) * 100;
    caidaRL = (resultados.(tam).bandit(1) - resultados.(tam).bandit) / resultados.(tam).bandit(1) * 100;
    plot(xVals, caidaSL, estiloEscala{v}, 'Color', colorSoftmax, 'LineWidth', grosorEscala(v), ...
        'MarkerFaceColor', colorSoftmax, 'MarkerSize', 6, 'DisplayName', sprintf('SL — %s', etiquetasVariante{v}));
    plot(xVals, caidaRL, estiloEscala{v}, 'Color', colorBandit, 'LineWidth', grosorEscala(v), ...
        'MarkerFaceColor', colorBandit, 'MarkerSize', 6, 'DisplayName', sprintf('RL — %s', etiquetasVariante{v}));
end
caidaSvm = (resultados.svm.acc(1) - resultados.svm.acc) / resultados.svm.acc(1) * 100;
caidaKnn = (resultados.knn.acc(1) - resultados.knn.acc) / resultados.knn.acc(1) * 100;
plot(xVals, caidaSvm, '-d', 'Color', colorSvm, 'LineWidth', 2.2, 'MarkerFaceColor', colorSvm, ...
    'MarkerSize', 7, 'DisplayName', 'SVM');
plot(xVals, caidaKnn, '-p', 'Color', colorKnn, 'LineWidth', 2.2, 'MarkerFaceColor', colorKnn, ...
    'MarkerSize', 8, 'DisplayName', 'k-NN (k=1)');
hold off;
grid on;
xticks(xVals); xticklabels(monthLabels);
ylabel('Caída relativa respecto a Mes0 (%)'); xlabel('Sesión');
title('Magnitud del drift: caída relativa de accuracy respecto a Mes0, los 5 modelos');
legend('Location', 'eastoutside', 'FontSize', 8);

exportgraphics(fig, fullfile(outDir, 'fig_drift_caida_relativa.png'), 'Resolution', 200);
close(fig);
fprintf('Guardado: %s\n', fullfile(outDir, 'fig_drift_caida_relativa.png'));

%% ======================================================================
%% FIGURA 4: fig_drift_vs_capacidad.png -- se mantiene backbone-only en X
%% (SVM/kNN no tienen "capacidad de backbone" comparable en parametros de
%% la misma familia; se anotan como lineas horizontales de referencia).
%% ======================================================================
fig = figure('Visible', 'off', 'Position', [100 100 1550 620]);

subplot(1,2,1);
m0SL = [resultados.pequeno.softmax(1), resultados.mediano.softmax(1), resultados.grande.softmax(1)];
m0RL = [resultados.pequeno.bandit(1), resultados.mediano.bandit(1), resultados.grande.bandit(1)];
paramsSorted = sort(numParams);
semilogx(paramsSorted, m0SL, '-o', 'Color', colorSoftmax, 'LineWidth', 2.2, 'MarkerFaceColor', colorSoftmax, 'MarkerSize', 8, 'DisplayName', 'SL'); hold on;
semilogx(paramsSorted, m0RL, '-o', 'Color', colorBandit, 'LineWidth', 2.2, 'MarkerFaceColor', colorBandit, 'MarkerSize', 8, 'DisplayName', 'RL');
yline(resultados.svm.acc(1), '--', 'Color', colorSvm, 'LineWidth', 1.6, 'DisplayName', 'SVM (sin backbone)');
yline(resultados.knn.acc(1), '--', 'Color', colorKnn, 'LineWidth', 1.6, 'DisplayName', 'k-NN (sin backbone)');
hold off; grid on;
xlabel('Parámetros del backbone (escala log)'); ylabel('Accuracy Mes0(val) (%)');
title('Capacidad del backbone vs. accuracy base');
legend('Location', 'southeast', 'FontSize', 8);
ylim([40 100]);

subplot(1,2,2);
dropSL = (m0SL - [resultados.pequeno.softmax(end), resultados.mediano.softmax(end), resultados.grande.softmax(end)]) ./ m0SL * 100;
dropRL = (m0RL - [resultados.pequeno.bandit(end), resultados.mediano.bandit(end), resultados.grande.bandit(end)]) ./ m0RL * 100;
dropSvm = (resultados.svm.acc(1) - resultados.svm.acc(end)) / resultados.svm.acc(1) * 100;
dropKnn = (resultados.knn.acc(1) - resultados.knn.acc(end)) / resultados.knn.acc(1) * 100;
semilogx(paramsSorted, dropSL, '-o', 'Color', colorSoftmax, 'LineWidth', 2.2, 'MarkerFaceColor', colorSoftmax, 'MarkerSize', 8, 'DisplayName', 'SL'); hold on;
semilogx(paramsSorted, dropRL, '-o', 'Color', colorBandit, 'LineWidth', 2.2, 'MarkerFaceColor', colorBandit, 'MarkerSize', 8, 'DisplayName', 'RL');
yline(dropSvm, '--', 'Color', colorSvm, 'LineWidth', 1.6, 'DisplayName', 'SVM (sin backbone)');
yline(dropKnn, '--', 'Color', colorKnn, 'LineWidth', 1.6, 'DisplayName', 'k-NN (sin backbone)');
hold off; grid on;
xlabel('Parámetros del backbone (escala log)'); ylabel('Caída relativa Mes0→Mes6 (%)');
title('Capacidad del backbone vs. magnitud del drift');
legend('Location', 'northwest', 'FontSize', 8);
ylim([0 30]);

sgtitle('¿El tamaño (o la ausencia) de un backbone aprendido protege contra el drift? (no de forma monótona)', ...
    'FontWeight', 'bold', 'FontSize', 13);

exportgraphics(fig, fullfile(outDir, 'fig_drift_vs_capacidad.png'), 'Resolution', 200);
close(fig);
fprintf('Guardado: %s\n', fullfile(outDir, 'fig_drift_vs_capacidad.png'));

fprintf('\n=== Resumen numerico (para verificar contra el texto del paper) ===\n');
fprintf('SVM:  M0=%.1f%%  M6=%.1f%%  drop=%.1f%%\n', resultados.svm.acc(1), resultados.svm.acc(end), caidaSvm(end));
fprintf('kNN:  M0=%.1f%%  M6=%.1f%%  drop=%.1f%%\n', resultados.knn.acc(1), resultados.knn.acc(end), caidaKnn(end));
