% #################################################################
% PARADIGMA C - LinUCB ADAPTATIVO (sigue aprendiendo Mes1..MesN)
% A diferencia de context_bandit.m (que se congela tras el warm-start de
% Mes0 y nunca mas se actualiza), este script parte del MISMO estado
% congelado -lo carga de Bandit_Results_*.mat, no repite el warm-start- y
% lo deja seguir aprendiendo online, mes a mes, usando UNICAMENTE la señal
% de recompensa derivada de la etiqueta real del gesto (+classWeights si
% acierta, -classWeights si falla) -- nunca usa las etiquetas para
% clasificar directamente, solo para calcular el reward, igual que en el
% warm-start.
%
% Protocolo "prequential" (test-then-train), estandar para evaluar
% aprendizaje online: en cada frame se predice PRIMERO con el estado
% actual (esa prediccion es la que cuenta para el accuracy del mes), y
% recien despues se actualiza A_inv/b con la recompensa de esa prediccion.
% Asi ningun frame contribuye a su propia prediccion.
%
% Solo procesa Mes1..Mes{Shared.NUM_VALID_TEST_MONTHS}: Mes5/Mes6 todavia
% no tienen groundTruthIndex (ver Shared.m), y usarlos como señal de
% recompensa ahora mismo corromperia el aprendizaje online.
%
% PARAMETRIZADO POR TAMAÑO (Tarea 2 / Experimento 2B): 'grande' (default),
% 'mediano', 'pequeno' -- busca el cache y el bandit congelado de ESE
% tamaño específicamente.
% #################################################################

function context_bandit_adaptive(tamano)

if nargin < 1
    tamano = 'grande';
end
tamano = validatestring(tamano, {'grande', 'mediano', 'pequeno', 'ann', 'tcn', 'micro', 'nano'});

clearvars -except tamano; clc;
rng(9);

%% CARGAR EL CACHE DE FEATURES MÁS RECIENTE DE ESTE TAMAÑO
cacheFiles = dir(fullfile('Models', sprintf('FrozenFeatures_%s_*.mat', tamano)));
if isempty(cacheFiles)
    error('No se encontró Models/FrozenFeatures_%s_*.mat. Corre buildFeatureCache(''%s'') primero.', tamano, tamano);
end
[~, newestIdx] = max([cacheFiles.datenum]);
cacheFile = fullfile(cacheFiles(newestIdx).folder, cacheFiles(newestIdx).name);
fprintf('Cache de features usado (%s): %s\n', tamano, cacheFile);
load(cacheFile, 'X_val', 'y_val', 'X_test', 'y_test', 'classNames');

%% CARGAR EL BANDIT YA CONGELADO DE ESTE TAMAÑO (warm-start de Mes0, de context_bandit.m)
banditFiles = dir(fullfile('Models', sprintf('Bandit_Results_%s_*.mat', tamano)));
if isempty(banditFiles)
    error('No se encontró Models/Bandit_Results_%s_*.mat. Corre context_bandit(''%s'') primero.', tamano, tamano);
end
[~, newestIdxB] = max([banditFiles.datenum]);
banditFile = fullfile(banditFiles(newestIdxB).folder, banditFiles(newestIdxB).name);
fprintf('Bandit congelado (punto de partida, %s) usado: %s\n', tamano, banditFile);
load(banditFile, 'A_inv', 'b', 'alpha', 'classWeights');

numClasses = numel(classNames);

%% BASELINE: desempeño ANTES de adaptar con Mes1..MesN (identico al de context_bandit.m)
predVal = predictBanditExploit(A_inv, b, X_val, classNames);

%% ADAPTACIÓN ONLINE: Mes1..MesN, prequential (predice y luego actualiza)
numMeses = Shared.NUM_VALID_TEST_MONTHS;
accByMonth = nan(1, numMeses + 1);
f1ByMonth = nan(1, numMeses + 1);
allMonthLabels = ["Mes0(val)", "Mes1", "Mes2", "Mes3", "Mes4", "Mes5", "Mes6"];
monthLabels = allMonthLabels(1:numMeses + 1);

[accByMonth(1), f1ByMonth(1)] = evaluateFold(y_val(:), predVal(:), classNames, 'BanditAdaptativo - Mes0(val)');
fprintf('Bandit adaptativo  Mes0(val) [punto de partida, sin adaptar aun]: accuracy=%.4f  macroF1=%.4f\n', accByMonth(1), f1ByMonth(1));

fprintf('\n========================================================================\n');
fprintf(' ADAPTACIÓN ONLINE MES1..MES%d (prequential: predice y luego aprende)\n', numMeses);
fprintf('========================================================================\n');

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
            V_inv = A_inv{i};
            recompensaEstimada = (V_inv * b{i})' * x_t;
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

        % Sherman-Morrison: actualizacion analitica de la inversa
        V = A_inv{brazoElegido};
        num = V * (x_t * x_t') * V;
        den = 1 + x_t' * V * x_t;
        A_inv{brazoElegido} = V - (num / den);
        b{brazoElegido} = b{brazoElegido} + (r_t * x_t);
    end

    predLabelsThisMonth = categorical(predLabelsThisMonth, classNames);
    [accByMonth(m + 1), f1ByMonth(m + 1)] = evaluateFold(ym(:), predLabelsThisMonth(:), classNames, sprintf('BanditAdaptativo - Mes%d', m));
    fprintf('Bandit adaptativo  Mes%d: accuracy=%.4f  macroF1=%.4f  (modelo actualizado con este mes antes de seguir)\n', m, accByMonth(m + 1), f1ByMonth(m + 1));
end

%% GUARDAR RESULTADOS (incluye el estado final adaptado, por si se quiere seguir alimentando)
if ~exist('Models', 'dir')
    mkdir('Models');
end
resultsFile = sprintf('Models/BanditAdaptive_Results_%s_%s.mat', tamano, datestr(now, 'dd-mm-yyyy_HH-MM-ss'));
save(resultsFile, 'A_inv', 'b', 'alpha', 'accByMonth', 'f1ByMonth', 'monthLabels', 'classNames', 'classWeights', 'cacheFile', 'banditFile', 'tamano');
fprintf('\nResultados guardados en: %s\n', resultsFile);

end


%% ======================================================
%% FUNCTION TO PREDICT WITH A FROZEN BANDIT (PURE EXPLOITATION)
%% Identica a la usada en context_bandit.m; solo se usa aqui para el punto
%% de partida (Mes0-val), antes de que empiece la adaptacion online.
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
%% FUNCTION TO EVALUATE ONE FOLD (accuracy, macro-F1, confusion matrix)
%% Identica a la usada en trainSoftmaxHead.m y context_bandit.m.
%% ======================================================
function [acc, macroF1] = evaluateFold(trueLabels, predLabels, classNames, foldName)
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

    figure('Name', ['Confusion Matrix - ' foldName]);
    matrixChart = confusionchart(confMat, classesCat);
    matrixChart.ColumnSummary = 'column-normalized';
    matrixChart.RowSummary = 'row-normalized';
    matrixChart.Title = ['Hand gestures - ' foldName];
    sortClasses(matrixChart, classesCat);
end
