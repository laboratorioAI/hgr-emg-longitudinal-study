% #################################################################
% Experimento adicional (2026-07-25, a raíz de la pregunta del usuario:
% "ayúdame a buscar alguna cuestión dónde se pueda determinar el drift"):
% mide si existe covariate shift REAL en el espacio de representación
% (las 128 características congeladas, capa dropout_2), independientemente
% de si el clasificador final logra compensarlo o no. Todos los
% experimentos anteriores (ver RESULTADOS_Y_VALIDACION.md §5.1-5.6)
% midieron drift indirectamente vía accuracy del clasificador -- este
% experimento mide el desplazamiento de la REPRESENTACIÓN directamente.
%
% Métodos usados (elegidos tras revisión bibliográfica, no arbitrarios):
%   1. MMD² (Maximum Mean Discrepancy, kernel RBF, bandwidth por "median
%      heuristic") + PRUEBA DE PERMUTACIÓN para obtener un p-value real
%      -- Gretton et al. (2012, JMLR), "A Kernel Two-Sample Test". El
%      estándar de facto en ML para esta pregunta exacta.
%   2. Domain classifier / A-distance (Ben-David et al. 2006/2010) --
%      entrena un clasificador binario "¿este frame es de Mes0 o de
%      MesN?" sobre las 128 features, con validación cruzada. Si logra
%      distinguirlos por encima del azar de forma significativa, hay
%      covariate shift medible en la representación misma.
% PCA/t-SNE se generan SOLO como figura de apoyo visual (no como
% evidencia rigurosa por sí solos -- ver discusión bibliográfica).
%
% Alcance: solo el backbone 'grande' (arquitectura principal), Mes0 vs.
% cada uno de Mes1..Mes6 por separado (6 comparaciones), para ver si el
% desplazamiento (si existe) crece progresivamente con el tiempo o es
% plano, igual que ya se observó con accuracy.
% #################################################################

clear; clc;
rng(9);

nSubsample = 2000; % frames por grupo para MMD/permutación (O(n^2) en el kernel -- se limita por costo computacional, no por conveniencia)
nPermutations = 2000; % iteraciones de la prueba de permutación
nFoldsDomainClassifier = 5;

%% CARGAR EL CACHE DE FEATURES (variante grande)
files = dir(fullfile('Models', 'FrozenFeatures_*.mat'));
esOtraVariante = contains({files.name}, {'_ann_', '_mediano_', '_pequeno_', '_tcn_'});
files = files(~esOtraVariante);
[~, idx] = max([files.datenum]);
cacheFile = fullfile(files(idx).folder, files(idx).name);
fprintf('Cache usado: %s\n', cacheFile);
load(cacheFile, 'X_train', 'X_val', 'X_test', 'classNames');

X_mes0 = [X_train, X_val]; % 128 x N0
numMeses = numel(X_test);
monthNames = arrayfun(@(m) sprintf('Mes%d', m), 1:numMeses, 'UniformOutput', false);

results = struct('mes', {}, 'mmd2', {}, 'pValuePermutacion', {}, ...
    'accDomainClassifier', {}, 'aDistance', {}, 'pValueDomainClassifier', {});

outDir = 'FigurasReales';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

%% BUCLE PRINCIPAL: Mes0 vs cada MesN
for m = 1:numMeses
    fprintf('\n========================================================================\n');
    fprintf(' Covariate shift: Mes0 vs Mes%d\n', m);
    fprintf('========================================================================\n');

    Xn = X_test{m};

    % Submuestreo reproducible (mismo tamaño en ambos grupos)
    rng(9 + m); % semilla distinta por mes pero reproducible, para no repetir siempre el mismo subconjunto de Mes0
    n0 = min(nSubsample, size(X_mes0, 2));
    nN = min(nSubsample, size(Xn, 2));
    idx0 = randperm(size(X_mes0, 2), n0);
    idxN = randperm(size(Xn, 2), nN);
    A = X_mes0(:, idx0)'; % n0 x 128
    B = Xn(:, idxN)';     % nN x 128

    %% 1. MMD^2 con kernel RBF (bandwidth = median heuristic) + permutación
    [mmd2Obs, sigma] = computeMMD2(A, B);
    fprintf('MMD^2 observado: %.6f (bandwidth RBF sigma=%.4f)\n', mmd2Obs, sigma);

    combined = [A; B];
    nA = size(A, 1);
    nTotal = size(combined, 1);
    mmd2Null = zeros(nPermutations, 1);
    for p = 1:nPermutations
        permIdx = randperm(nTotal);
        Ap = combined(permIdx(1:nA), :);
        Bp = combined(permIdx(nA+1:end), :);
        mmd2Null(p) = computeMMD2(Ap, Bp, sigma); % mismo sigma que la observación, para que la prueba sea justa
    end
    pValueMMD = mean(mmd2Null >= mmd2Obs);
    fprintf('Prueba de permutación (%d iteraciones): p-value = %.4f\n', nPermutations, pValueMMD);

    %% 2. Domain classifier (A-distance), validación cruzada
    domainLabels = [zeros(nA, 1); ones(size(B, 1), 1)];
    domainFeatures = [A; B];
    cv = cvpartition(domainLabels, 'KFold', nFoldsDomainClassifier);
    accFolds = zeros(nFoldsDomainClassifier, 1);
    for f = 1:nFoldsDomainClassifier
        trainIdx = training(cv, f);
        testIdx = test(cv, f);
        mdl = fitclinear(domainFeatures(trainIdx, :), domainLabels(trainIdx), 'Learner', 'logistic');
        predFold = predict(mdl, domainFeatures(testIdx, :));
        accFolds(f) = mean(predFold == domainLabels(testIdx));
    end
    accDomain = mean(accFolds);
    epsilonError = 1 - accDomain;
    aDistance = 2 * (1 - 2 * epsilonError); % Ben-David et al.: A-distance = 2(1-2*error)
    fprintf('Domain classifier accuracy (5-fold CV): %.4f  ->  A-distance = %.4f\n', accDomain, aDistance);

    % p-value del domain classifier: permutación de etiquetas de dominio,
    % reentrenando el clasificador cada vez (más caro, se usa un numero
    % menor de permutaciones que para MMD por costo computacional)
    nPermDomain = 200;
    accNullDomain = zeros(nPermDomain, 1);
    for p = 1:nPermDomain
        shuffledLabels = domainLabels(randperm(numel(domainLabels)));
        cvp = cvpartition(shuffledLabels, 'KFold', nFoldsDomainClassifier);
        accP = zeros(nFoldsDomainClassifier, 1);
        for f = 1:nFoldsDomainClassifier
            trainIdx = training(cvp, f);
            testIdx = test(cvp, f);
            mdl = fitclinear(domainFeatures(trainIdx, :), shuffledLabels(trainIdx), 'Learner', 'logistic');
            predFold = predict(mdl, domainFeatures(testIdx, :));
            accP(f) = mean(predFold == shuffledLabels(testIdx));
        end
        accNullDomain(p) = mean(accP);
    end
    pValueDomain = mean(accNullDomain >= accDomain);
    fprintf('Prueba de permutación del domain classifier (%d iteraciones): p-value = %.4f\n', nPermDomain, pValueDomain);

    results(m).mes = m;
    results(m).mmd2 = mmd2Obs;
    results(m).pValuePermutacion = pValueMMD;
    results(m).accDomainClassifier = accDomain;
    results(m).aDistance = aDistance;
    results(m).pValueDomainClassifier = pValueDomain;

    %% FIGURA DE APOYO: PCA 2D, Mes0 vs MesN (solo ilustrativo)
    [~, scoreA] = pca([A; B]);
    fig = figure('Visible', 'off', 'Position', [100 100 700 600]);
    hold on;
    scatter(scoreA(1:nA, 1), scoreA(1:nA, 2), 8, [0.20 0.35 0.60], 'filled', 'MarkerFaceAlpha', 0.4, 'DisplayName', 'Mes0');
    scatter(scoreA(nA+1:end, 1), scoreA(nA+1:end, 2), 8, [0.85 0.33 0.10], 'filled', 'MarkerFaceAlpha', 0.4, 'DisplayName', sprintf('Mes%d', m));
    hold off;
    xlabel('PC1'); ylabel('PC2');
    title(sprintf('PCA de features (128-d) — Mes0 vs Mes%d (solo ilustrativo, ver MMD/A-distance para rigor)', m));
    legend('Location', 'best');
    grid on;
    exportgraphics(fig, fullfile(outDir, sprintf('covshift_pca_mes0_vs_mes%d.png', m)), 'Resolution', 150);
    close(fig);
end

%% GUARDAR RESULTADOS
if ~exist('Models', 'dir')
    mkdir('Models');
end
resultsFile = sprintf('Models/CovariateShift_%s.mat', datestr(now, 'dd-mm-yyyy_HH-MM-ss'));
save(resultsFile, 'results', 'monthNames', 'nSubsample', 'nPermutations', 'cacheFile');
fprintf('\nResultados guardados en: %s\n', resultsFile);

%% TABLA RESUMEN
fprintf('\n=====================================================================================\n');
fprintf(' RESUMEN: COVARIATE SHIFT EN EL ESPACIO DE REPRESENTACIÓN (128-d), Mes0 vs MesN\n');
fprintf('=====================================================================================\n');
fprintf('%-8s %12s %12s %14s %12s %14s\n', 'Mes', 'MMD^2', 'p(MMD)', 'AccDomainClf', 'A-distance', 'p(DomainClf)');
for m = 1:numMeses
    fprintf('%-8s %12.6f %12.4f %14.4f %12.4f %14.4f\n', monthNames{m}, results(m).mmd2, ...
        results(m).pValuePermutacion, results(m).accDomainClassifier, results(m).aDistance, results(m).pValueDomainClassifier);
end

%% FIGURA: MMD^2 y A-distance por mes, con marcadores de significancia
% Paneles apilados verticalmente (uno arriba del otro), no lado a lado --
% pensado para encajar en una sola columna de un documento IEEE a doble
% columna, donde un layout de 2 paneles horizontales queda demasiado
% angosto y apretado para leerse bien.
fig2 = figure('Visible', 'off', 'Position', [100 100 500 700]);
subplot(2, 1, 1);
mmdVals = [results.mmd2];
pVals = [results.pValuePermutacion];
bar(1:numMeses, mmdVals, 'FaceColor', [0.34 0.61 0.84]);
hold on;
for m = 1:numMeses
    if pVals(m) < 0.05
        text(m, mmdVals(m), '*', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 16, 'FontWeight', 'bold');
    end
end
hold off;
xticks(1:numMeses); xticklabels(monthNames);
ylabel('MMD^2 (Mes0 vs MesN)');
title('MMD^2 (* = p<0.05, prueba de permutación)');
grid on;

subplot(2, 1, 2);
aDistVals = [results.aDistance];
pValsDomain = [results.pValueDomainClassifier];
bar(1:numMeses, aDistVals, 'FaceColor', [0.93 0.49 0.19]);
hold on;
for m = 1:numMeses
    if pValsDomain(m) < 0.05
        text(m, aDistVals(m), '*', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 16, 'FontWeight', 'bold');
    end
end
hold off;
xticks(1:numMeses); xticklabels(monthNames);
ylabel('A-distance (0=idénticos, 2=totalmente distinguibles)');
title('A-distance / domain classifier (* = p<0.05)');
grid on;

sgtitle('Covariate shift en el espacio de representación (128-d)', 'FontSize', 11);
exportgraphics(fig2, fullfile(outDir, 'covariate_shift_resumen.png'), 'Resolution', 200);
close(fig2);

fprintf('\nFiguras guardadas en: %s\n', fullfile(pwd, outDir));


%% ======================================================
%% FUNCTION: MMD^2 con kernel RBF (median heuristic para el bandwidth)
%% ======================================================
function [mmd2, sigma] = computeMMD2(A, B, sigmaFixed)
    % A: nA x d, B: nB x d. MMD^2 = E[k(a,a')] + E[k(b,b')] - 2*E[k(a,b)]
    % con kernel RBF k(x,y) = exp(-||x-y||^2 / (2*sigma^2)).
    if nargin < 3 || isempty(sigmaFixed)
        % Median heuristic: sigma = mediana de las distancias euclidianas
        % por pares en la unión de A y B (Gretton et al. 2012).
        combined = [A; B];
        nSample = min(500, size(combined, 1)); % submuestra para que el calculo de la mediana no sea O(n^2) completo
        subIdx = randperm(size(combined, 1), nSample);
        D = pdist(combined(subIdx, :));
        sigma = median(D);
        if sigma == 0
            sigma = 1; % salvaguarda si todos los puntos submuestreados coinciden (no debería ocurrir con datos reales)
        end
    else
        sigma = sigmaFixed;
    end

    Kaa = rbfKernel(A, A, sigma);
    Kbb = rbfKernel(B, B, sigma);
    Kab = rbfKernel(A, B, sigma);

    nA = size(A, 1);
    nB = size(B, 1);
    % Se excluye la diagonal (k(a,a)=1 siempre) para el estimador insesgado estándar
    termAA = (sum(Kaa(:)) - nA) / (nA * (nA - 1));
    termBB = (sum(Kbb(:)) - nB) / (nB * (nB - 1));
    termAB = sum(Kab(:)) / (nA * nB);

    mmd2 = termAA + termBB - 2 * termAB;
end

function K = rbfKernel(X, Y, sigma)
    D2 = pdist2(X, Y, 'squaredeuclidean');
    K = exp(-D2 / (2 * sigma^2));
end
