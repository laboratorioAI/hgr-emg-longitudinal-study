% #################################################################
% Figuras del experimento de modelos personalizados (2026-08-11): 19
% modelos independientes por arquitectura (SVM lineal C=10, kNN k=9
% coseno, Softmax sobre backbone 'pequeno'), uno por usuario, evaluados
% Mes0(val)->Mes6. Objetivo de las figuras: mostrar que el modelo
% poblacional (promedio de los 19 usuarios mezclados) esconde una
% variabilidad inter-usuario muy grande, que solo es visible entrenando
% un modelo por persona.
% #################################################################

clear; clc;

baseDir = 'Models/';
fSvm = dir(fullfile(baseDir, 'PersonalizedModels_svm_linear_C10_*.mat'));
[~, idx] = max([fSvm.datenum]);
sSvm = load(fullfile(fSvm(idx).folder, fSvm(idx).name));

fKnn = dir(fullfile(baseDir, 'PersonalizedModels_knn_k9cosine_*.mat'));
[~, idx] = max([fKnn.datenum]);
sKnn = load(fullfile(fKnn(idx).folder, fKnn(idx).name));

haveSoftmax = false;
fSm = dir(fullfile(baseDir, 'PersonalizedModels_softmax_pequeno_*.mat'));
if ~isempty(fSm)
    [~, idx] = max([fSm.datenum]);
    sSm = load(fullfile(fSm(idx).folder, fSm(idx).name));
    haveSoftmax = true;
end

monthLabels = cellstr(sSvm.monthLabels);
monthLabels = strrep(monthLabels, 'Mes0(val)', 'Month0(val)');
monthLabels = regexprep(monthLabels, '^Mes(\d+)$', 'Month$1');
users = sSvm.allUsers;
numUsers = numel(users);
numMonths = numel(monthLabels);

% Matrices usuario x mes
matSvm = nan(numUsers, numMonths);
matKnn = nan(numUsers, numMonths);
for u = 1:numUsers
    matSvm(u,:) = sSvm.resultadosSVM.(users(u)).accByMonth * 100;
    matKnn(u,:) = sKnn.resultadosKNN.(users(u)).accByMonth * 100;
end
if haveSoftmax
    matSm = nan(numUsers, numMonths);
    for u = 1:numUsers
        matSm(u,:) = sSm.resultadosSoftmax.(users(u)).accByMonth * 100;
    end
end

outDir = 'FigurasReales';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

colorUser = lines(numUsers);

%% ======================================================================
%% FIGURA 1: lineas individuales por usuario, un panel por modelo
%% ======================================================================
if haveSoftmax
    fig = figure('Visible', 'off', 'Position', [50 50 1900 650]);
    nPanels = 3;
    paneles = {matSvm, matKnn, matSm};
    titulos = {'SVM lineal (C=10)', 'k-NN (k=9, coseno)', 'Softmax (backbone pequeño)'};
else
    fig = figure('Visible', 'off', 'Position', [50 50 1300 650]);
    nPanels = 2;
    paneles = {matSvm, matKnn};
    titulos = {'SVM lineal (C=10)', 'k-NN (k=9, coseno)'};
end

for p = 1:nPanels
    subplot(1, nPanels, p);
    hold on;
    for u = 1:numUsers
        plot(1:numMonths, paneles{p}(u,:), '-o', 'Color', [colorUser(u,:) 0.55], ...
            'LineWidth', 1.1, 'MarkerSize', 3, 'MarkerFaceColor', colorUser(u,:));
    end
    meanCurve = mean(paneles{p}, 1, 'omitnan');
    plot(1:numMonths, meanCurve, '-o', 'Color', 'k', 'LineWidth', 3, 'MarkerSize', 6, 'MarkerFaceColor', 'k');
    hold off;
    grid on;
    xticks(1:numMonths); xticklabels(monthLabels);
    ylim([0 100]);
    ylabel('Accuracy (%)'); xlabel('Sesión');
    title(titulos{p});
    if p == 1
        text(1.1, 8, 'Línea negra gruesa = promedio poblacional (19 modelos)', 'FontSize', 8, 'Color', [0.3 0.3 0.3]);
    end
end
sgtitle('Modelos personalizados: cada línea de color es UN usuario con SU PROPIO modelo entrenado solo con sus datos', ...
    'FontWeight', 'bold', 'FontSize', 13);
exportgraphics(fig, fullfile(outDir, 'fig_personalizado_lineas_por_usuario.png'), 'Resolution', 200);
close(fig);
fprintf('Guardado: fig_personalizado_lineas_por_usuario.png\n');

%% ======================================================================
%% FIGURA 2: heatmap usuario x mes (uno por modelo)
%% ======================================================================
if haveSoftmax
    fig = figure('Visible', 'off', 'Position', [50 50 1900 700]);
else
    fig = figure('Visible', 'off', 'Position', [50 50 1300 700]);
end
for p = 1:nPanels
    subplot(1, nPanels, p);
    imagesc(paneles{p});
    colormap(gca, turbo);
    clim([0 100]);
    colorbar;
    xticks(1:numMonths); xticklabels(monthLabels); xtickangle(45);
    yticks(1:numUsers); yticklabels(cellstr(users));
    title(titulos{p});
    xlabel('Sesión'); ylabel('Usuario');
    for u = 1:numUsers
        for m = 1:numMonths
            text(m, u, sprintf('%.0f', paneles{p}(u,m)), 'HorizontalAlignment', 'center', ...
                'FontSize', 6.5, 'Color', [0.1 0.1 0.1]);
        end
    end
end
sgtitle('Heatmap usuario × sesión: accuracy (%) de cada modelo personalizado', 'FontWeight', 'bold', 'FontSize', 13);
exportgraphics(fig, fullfile(outDir, 'fig_personalizado_heatmap.png'), 'Resolution', 200);
close(fig);
fprintf('Guardado: fig_personalizado_heatmap.png\n');

%% ======================================================================
%% FIGURA 3: poblacional vs. personalizado -- media +/- SD por mes
%% ======================================================================
fig = figure('Visible', 'off', 'Position', [50 50 1300 650]);
hold on;
colorSvm = [0.20 0.55 0.20];
colorKnn = [0.55 0.25 0.65];
colorSm = [0.85 0.33 0.10];

meanSvm = mean(matSvm, 1, 'omitnan'); sdSvm = std(matSvm, 0, 1, 'omitnan');
meanKnn = mean(matKnn, 1, 'omitnan'); sdKnn = std(matKnn, 0, 1, 'omitnan');

errorbar(1:numMonths, meanSvm, sdSvm, '-o', 'Color', colorSvm, 'LineWidth', 2.2, ...
    'MarkerFaceColor', colorSvm, 'MarkerSize', 7, 'CapSize', 6, 'DisplayName', 'SVM lineal — media ± SD entre usuarios');
errorbar(1:numMonths, meanKnn, sdKnn, '-s', 'Color', colorKnn, 'LineWidth', 2.2, ...
    'MarkerFaceColor', colorKnn, 'MarkerSize', 7, 'CapSize', 6, 'DisplayName', 'k-NN k=9 — media ± SD entre usuarios');
if haveSoftmax
    meanSm = mean(matSm, 1, 'omitnan'); sdSm = std(matSm, 0, 1, 'omitnan');
    errorbar(1:numMonths, meanSm, sdSm, '-^', 'Color', colorSm, 'LineWidth', 2.2, ...
        'MarkerFaceColor', colorSm, 'MarkerSize', 7, 'CapSize', 6, 'DisplayName', 'Softmax (pequeño) — media ± SD entre usuarios');
end
hold off;
grid on;
xticks(1:numMonths); xticklabels(monthLabels);
ylabel('Accuracy (%)'); xlabel('Sesión');
ylim([0 100]);
title('Variabilidad entre usuarios: la banda de SD muestra lo que el promedio poblacional esconde');
legend('Location', 'southoutside', 'NumColumns', 1, 'FontSize', 9);
exportgraphics(fig, fullfile(outDir, 'fig_personalizado_media_sd.png'), 'Resolution', 200);
close(fig);
fprintf('Guardado: fig_personalizado_media_sd.png\n');

%% ======================================================================
%% FIGURA 4: SD entre usuarios por mes (una sola metrica de dispersion)
%% ======================================================================
fig = figure('Visible', 'off', 'Position', [50 50 1100 600]);
hold on;
plot(1:numMonths, sdSvm, '-o', 'Color', colorSvm, 'LineWidth', 2.2, 'MarkerFaceColor', colorSvm, 'MarkerSize', 7, 'DisplayName', 'SVM (linear)');
plot(1:numMonths, sdKnn, '-s', 'Color', colorKnn, 'LineWidth', 2.2, 'MarkerFaceColor', colorKnn, 'MarkerSize', 7, 'DisplayName', 'k-NN (k=9)');
if haveSoftmax
    plot(1:numMonths, sdSm, '-^', 'Color', colorSm, 'LineWidth', 2.2, 'MarkerFaceColor', colorSm, 'MarkerSize', 7, 'DisplayName', 'Softmax (small)');
end
hold off;
grid on;
xticks(1:numMonths); xticklabels(monthLabels);
ylabel('Between-participant SD (accuracy points)'); xlabel('Session');
title('Between-participant dispersion by session');
legend('Location', 'northwest', 'FontSize', 9);
exportgraphics(fig, fullfile(outDir, 'fig_personalizado_sd_por_mes_en.png'), 'Resolution', 200);
close(fig);
fprintf('Saved: fig_personalizado_sd_por_mes_en.png\n');
return; % ICASSP-paper variant only needs this SD-by-month figure

%% ======================================================================
%% RESUMEN NUMERICO
%% ======================================================================
fprintf('\n=== Resumen: media +/- SD entre los %d usuarios, por mes ===\n', numUsers);
fprintf('%-8s', 'Mes');
for m = 1:numMonths
    fprintf('%12s', monthLabels{m});
end
fprintf('\n');
fprintf('%-8s', 'SVM');
for m = 1:numMonths
    fprintf('%7.1f±%3.1f', meanSvm(m), sdSvm(m));
end
fprintf('\n');
fprintf('%-8s', 'kNN');
for m = 1:numMonths
    fprintf('%7.1f±%3.1f', meanKnn(m), sdKnn(m));
end
fprintf('\n');
if haveSoftmax
    fprintf('%-8s', 'Softmax');
    for m = 1:numMonths
        fprintf('%7.1f±%3.1f', meanSm(m), sdSm(m));
    end
    fprintf('\n');
end

fprintf('\nFiguras de modelos personalizados completadas.\n');
