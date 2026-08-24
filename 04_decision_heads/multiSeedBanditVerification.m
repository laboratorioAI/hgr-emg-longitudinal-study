% #################################################################
% VERIFICACION MULTISEMILLA - Paradigma B (LinUCB congelado) vs.
% Paradigma C (LinUCB adaptativo)
%
% Repite exactamente la misma logica de context_bandit.m (warm-start +
% Sherman-Morrison + congelado) seguida de context_bandit_adaptive.m
% (adaptacion online prequential), para varias semillas distintas, sobre
% el MISMO cache de features congeladas (no se regenera nada de datos).
%
% Objetivo: el patron "el bandit adaptativo tiene macro-F1 mayor que el
% congelado en todos los meses" (visto el 2026-07-15 con NUM_VALID_TEST_MONTHS=6)
% se observo con una sola semilla (rng(9), fija en ambos scripts). Aqui se
% varia esa semilla para ver si el patron es robusto o es sensible a los
% desempates aleatorios (randi) y al orden de barajado (randperm) del
% bandit durante el warm-start y la adaptacion online.
%
% No modifica context_bandit.m ni context_bandit_adaptive.m, ni genera
% Bandit_Results_*.mat / BanditAdaptive_Results_*.mat por semilla (para no
% interferir con la auto-deteccion "mas reciente" que usan los demas
% scripts) - todo se mantiene en memoria hasta el resumen final.
% #################################################################

clear; clc;

seeds = [1 2 3 4 5];

%% CARGAR EL CACHE DE FEATURES MAS RECIENTE (fijo para todas las semillas)
cacheFiles = dir(fullfile('Models', 'FrozenFeatures_*.mat'));
if isempty(cacheFiles)
    error('No se encontro Models/FrozenFeatures_*.mat. Corre buildFeatureCache.m primero.');
end
[~, newestIdx] = max([cacheFiles.datenum]);
cacheFile = fullfile(cacheFiles(newestIdx).folder, cacheFiles(newestIdx).name);
fprintf('Cache de features usado (fijo para todas las semillas): %s\n', cacheFile);
load(cacheFile, 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', 'classNames');

numClasses = numel(classNames);
d = size(X_train, 1);

%% PESOS DE CLASE (identico a context_bandit.m)
counts = countcats(y_train(:));
classWeights = sum(counts) ./ (numClasses * counts);

%% PARAMETROS (identicos a context_bandit.m / context_bandit_adaptive.m)
alpha = 0.5;
maxEpochs = 20;
patience = 5;

numMeses = Shared.NUM_VALID_TEST_MONTHS;
allMonthLabels = ["Mes0(val)", "Mes1", "Mes2", "Mes3", "Mes4", "Mes5", "Mes6"];
monthLabels = allMonthLabels(1:numMeses + 1);

numSeeds = numel(seeds);
accFrozenAll = nan(numSeeds, numMeses + 1);
f1FrozenAll  = nan(numSeeds, numMeses + 1);
accAdaptAll  = nan(numSeeds, numMeses + 1);
f1AdaptAll   = nan(numSeeds, numMeses + 1);

for s = 1:numSeeds
    seed = seeds(s);
    fprintf('\n========================================================================\n');
    fprintf(' SEMILLA %d (%d/%d)\n', seed, s, numSeeds);
    fprintf('========================================================================\n');
    rng(seed);

    %% --- WARM-START SOBRE MES0 (identico a context_bandit.m) ---
    A_inv = cell(numClasses, 1);
    b = cell(numClasses, 1);
    for i = 1:numClasses
        A_inv{i} = eye(d);
        b{i} = zeros(d, 1);
    end

    Ntrain = size(X_train, 2);
    bestValAcc = -Inf;
    bestA_inv = A_inv;
    bestB = b;
    epochsSinceImprovement = 0;

    for ep = 1:maxEpochs
        order = randperm(Ntrain);
        for k = 1:Ntrain
            t = order(k);
            x_t = X_train(:, t);
            realIdx = double(y_train(t));

            p_t = zeros(numClasses, 1);
            for i = 1:numClasses
                V_inv = A_inv{i};
                recompensaEstimada = (V_inv * b{i})' * x_t;
                incertidumbre = alpha * sqrt(max(x_t' * V_inv * x_t, 0));
                p_t(i) = recompensaEstimada + incertidumbre;
            end
            candidatos = find(p_t >= max(p_t) - 1e-9);
            brazoElegido = candidatos(randi(numel(candidatos)));

            if brazoElegido == realIdx
                r_t = classWeights(realIdx);
            else
                r_t = -classWeights(realIdx);
            end

            V = A_inv{brazoElegido};
            num = V * (x_t * x_t') * V;
            den = 1 + x_t' * V * x_t;
            A_inv{brazoElegido} = V - (num / den);
            b{brazoElegido} = b{brazoElegido} + (r_t * x_t);
        end

        predVal = predictBanditExploit(A_inv, b, X_val, classNames);
        valAcc = mean(predVal(:) == y_val(:));

        if valAcc > bestValAcc
            bestValAcc = valAcc;
            bestA_inv = A_inv;
            bestB = b;
            epochsSinceImprovement = 0;
        else
            epochsSinceImprovement = epochsSinceImprovement + 1;
            if epochsSinceImprovement >= patience
                break;
            end
        end
    end

    fprintf('  Warm-start: mejor accuracy Mes0(val) = %.4f\n', bestValAcc);

    %% --- CONGELADO: evaluar Mes0(val) + Mes1..MesN (identico a context_bandit.m) ---
    A_inv_frozen = bestA_inv;
    b_frozen = bestB;

    predVal = predictBanditExploit(A_inv_frozen, b_frozen, X_val, classNames);
    [accFrozenAll(s, 1), f1FrozenAll(s, 1)] = evaluateFoldQuiet(y_val(:), predVal(:), classNames);

    for m = 1:numMeses
        predTest = predictBanditExploit(A_inv_frozen, b_frozen, X_test{m}, classNames);
        [accFrozenAll(s, m + 1), f1FrozenAll(s, m + 1)] = evaluateFoldQuiet(y_test{m}(:), predTest(:), classNames);
    end

    fprintf('  Congelado   Mes0(val)=%.4f', accFrozenAll(s, 1));
    for m = 1:numMeses
        fprintf('  Mes%d=%.4f', m, accFrozenAll(s, m + 1));
    end
    fprintf('\n');

    %% --- ADAPTATIVO: parte del MISMO estado congelado de esta semilla (identico a context_bandit_adaptive.m) ---
    A_inv_adapt = bestA_inv;
    b_adapt = bestB;

    accAdaptAll(s, 1) = accFrozenAll(s, 1); % mismo punto de partida, antes de adaptar
    f1AdaptAll(s, 1) = f1FrozenAll(s, 1);

    for m = 1:numMeses
        Xm = X_test{m};
        ym = y_test{m};
        Nm = size(Xm, 2);
        predLabelsThisMonth = strings(1, Nm);

        for t = 1:Nm
            x_t = Xm(:, t);
            realIdx = double(ym(t));

            p_t = zeros(numClasses, 1);
            for i = 1:numClasses
                V_inv = A_inv_adapt{i};
                recompensaEstimada = (V_inv * b_adapt{i})' * x_t;
                incertidumbre = alpha * sqrt(max(x_t' * V_inv * x_t, 0));
                p_t(i) = recompensaEstimada + incertidumbre;
            end
            candidatos = find(p_t >= max(p_t) - 1e-9);
            brazoElegido = candidatos(randi(numel(candidatos)));
            predLabelsThisMonth(t) = classNames(brazoElegido);

            if brazoElegido == realIdx
                r_t = classWeights(realIdx);
            else
                r_t = -classWeights(realIdx);
            end

            V = A_inv_adapt{brazoElegido};
            num = V * (x_t * x_t') * V;
            den = 1 + x_t' * V * x_t;
            A_inv_adapt{brazoElegido} = V - (num / den);
            b_adapt{brazoElegido} = b_adapt{brazoElegido} + (r_t * x_t);
        end

        predLabelsThisMonth = categorical(predLabelsThisMonth, classNames);
        [accAdaptAll(s, m + 1), f1AdaptAll(s, m + 1)] = evaluateFoldQuiet(ym(:), predLabelsThisMonth(:), classNames);
    end

    fprintf('  Adaptativo  Mes0(val)=%.4f', accAdaptAll(s, 1));
    for m = 1:numMeses
        fprintf('  Mes%d=%.4f', m, accAdaptAll(s, m + 1));
    end
    fprintf('\n');
end

%% RESUMEN: media +/- desviacion estandar entre semillas, por mes y paradigma
fprintf('\n========================================================================\n');
fprintf(' RESUMEN MULTISEMILLA (%d semillas: %s)\n', numSeeds, mat2str(seeds));
fprintf('========================================================================\n');
fprintf('%-10s | %-22s | %-22s | %s\n', 'Mes', 'Congelado acc (m+-sd)', 'Adaptativo acc (m+-sd)', 'Adaptativo gana en');
for mIdx = 1:(numMeses + 1)
    accF = accFrozenAll(:, mIdx);
    accA = accAdaptAll(:, mIdx);
    diffs = accA - accF;
    numWins = sum(diffs > 0);
    fprintf('%-10s | %6.4f +- %6.4f      | %6.4f +- %6.4f      | %d/%d semillas\n', ...
        monthLabels(mIdx), mean(accF), std(accF), mean(accA), std(accA), numWins, numSeeds);
end

fprintf('\n%-10s | %-22s | %-22s\n', 'Mes', 'Congelado F1 (m+-sd)', 'Adaptativo F1 (m+-sd)');
for mIdx = 1:(numMeses + 1)
    f1F = f1FrozenAll(:, mIdx);
    f1A = f1AdaptAll(:, mIdx);
    fprintf('%-10s | %6.4f +- %6.4f      | %6.4f +- %6.4f\n', ...
        monthLabels(mIdx), mean(f1F), std(f1F), mean(f1A), std(f1A));
end

%% GUARDAR RESULTADOS CONSOLIDADOS
if ~exist('Models', 'dir')
    mkdir('Models');
end
resultsFile = sprintf('Models/MultiSeedBanditVerification_%s.mat', datestr(now, 'dd-mm-yyyy_HH-MM-ss'));
save(resultsFile, 'seeds', 'monthLabels', 'accFrozenAll', 'f1FrozenAll', 'accAdaptAll', 'f1AdaptAll', ...
    'cacheFile', 'alpha', 'maxEpochs', 'patience');
fprintf('\nResultados guardados en: %s\n', resultsFile);


%% ======================================================
%% FUNCTION TO PREDICT WITH A FROZEN BANDIT (PURE EXPLOITATION)
%% Identica a la usada en context_bandit.m / context_bandit_adaptive.m
%% ======================================================
function predLabels = predictBanditExploit(A_inv, b, X, classNames)
    numClasses = numel(classNames);
    theta = zeros(size(X, 1), numClasses);
    for i = 1:numClasses
        theta(:, i) = A_inv{i} * b{i};
    end
    scores = theta' * X;
    [~, armIdx] = max(scores, [], 1);
    predLabels = categorical(classNames(armIdx), classNames);
end

%% ======================================================
%% FUNCTION TO EVALUATE ONE FOLD (accuracy, macro-F1) - sin generar figuras
%% (a diferencia de evaluateFold en context_bandit.m/context_bandit_adaptive.m,
%% aqui se corren decenas de folds por semilla y no tiene sentido abrir una
%% figura de matriz de confusion por cada uno)
%% ======================================================
function [acc, macroF1] = evaluateFoldQuiet(trueLabels, predLabels, classNames)
    classesCat = categorical(classNames);
    confMat = confusionmat(trueLabels, predLabels, 'Order', classesCat);
    acc = sum(diag(confMat)) / sum(confMat(:));

    numClasses = numel(classNames);
    f1 = zeros(numClasses, 1);
    for c = 1:numClasses
        tp = confMat(c, c);
        fp = sum(confMat(:, c)) - tp;
        fn = sum(confMat(c, :)) - tp;
        precision = tp / max(tp + fp, 1);
        recall = tp / max(tp + fn, 1);
        if (precision + recall) == 0
            f1(c) = 0;
        else
            f1(c) = 2 * precision * recall / (precision + recall);
        end
    end
    macroF1 = mean(f1);
end
