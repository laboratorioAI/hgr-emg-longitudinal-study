% #################################################################
% PASO 2c: PARADIGMA B - LinUCB SOBRE EL BACKBONE CONGELADO
% Reescrito por completo (la versión anterior cargaba un modelo/datastore
% de una generación de datos obsoleta, mezclaba en el score del bandit la
% predicción softmax de otra red, y nunca evaluaba por mes).
%
% Usa el mismo cache de features congeladas (dropout_2, 128-d) que
% trainSoftmaxHead.m, para que ambos paradigmas partan exactamente de la
% misma información. El bandit se "entrena" (warm-start) una sola vez con
% Mes0 (training), con early stopping sobre Mes0 (validation) -mismo
% criterio de parada que usa trainSoftmaxHead.m-, se CONGELA, y se evalúa
% sin más actualizaciones en Mes0(validation, baseline) y Mes1..Mes6. Así
% la comparación contra el softmax es justa: ninguno de los dos aprende
% durante la evaluación temporal.
%
% PARAMETRIZADO POR TAMAÑO (Tarea 2 / Experimento 2B): 'grande' (default),
% 'mediano', 'pequeno' -- busca el cache de ESE tamaño específicamente.
% #################################################################

function context_bandit(tamano)

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
load(cacheFile, 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', 'classNames');

numClasses = numel(classNames);
d = size(X_train, 1);

%% PESOS DE CLASE (misma fórmula que trainSoftmaxHead.m / backbone: usados
%% aquí como magnitud de la recompensa, en vez de constantes arbitrarias)
counts = countcats(y_train(:));
classWeights = sum(counts) ./ (numClasses * counts);
fprintf('\nPesos de clase (bandit, derivados de Mes0 training, usados como |recompensa|):\n');
for i = 1:numClasses
    fprintf('  %-10s peso=%.4f\n', classNames(i), classWeights(i));
end

%% PARÁMETROS DEL EXPERIMENTO
alpha = 0.5;          % escala del término de incertidumbre UCB (solo durante warm-start)
maxEpochs = 20;       % máximo de pasadas sobre Mes0 training
patience = 5;         % igual criterio de parada temprana que ValidationPatience del softmax

%% INICIALIZACIÓN (Sherman-Morrison: se guarda directamente la inversa)
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

fprintf('\n========================================================================\n');
fprintf(' WARM-START LinUCB SOBRE MES0 (training), early stopping sobre Mes0 (validation)\n');
fprintf('========================================================================\n');

for ep = 1:maxEpochs
    order = randperm(Ntrain); % 'Shuffle every-epoch', igual que el softmax

    for k = 1:Ntrain
        t = order(k);
        x_t = X_train(:, t);
        realIdx = double(y_train(t)); % el código categórico ya sigue el orden de classNames

        p_t = zeros(numClasses, 1);
        for i = 1:numClasses
            V_inv = A_inv{i};
            recompensaEstimada = (V_inv * b{i})' * x_t;
            incertidumbre = alpha * sqrt(max(x_t' * V_inv * x_t, 0));
            p_t(i) = recompensaEstimada + incertidumbre;
        end
        % Empates (p.ej. al inicio, con A_inv/b identicos entre brazos) se
        % resuelven al azar; max() por si solo siempre elegiria el indice
        % mas bajo, sesgando la exploracion temprana hacia esa clase.
        candidatos = find(p_t >= max(p_t) - 1e-9);
        brazoElegido = candidatos(randi(numel(candidatos)));

        if brazoElegido == realIdx
            r_t = classWeights(realIdx);
        else
            r_t = -classWeights(realIdx);
        end

        % Sherman-Morrison: actualización analítica de la inversa
        V = A_inv{brazoElegido};
        num = V * (x_t * x_t') * V;
        den = 1 + x_t' * V * x_t;
        A_inv{brazoElegido} = V - (num / den);
        b{brazoElegido} = b{brazoElegido} + (r_t * x_t);
    end

    predVal = predictBanditExploit(A_inv, b, X_val, classNames);
    valAcc = mean(predVal(:) == y_val(:));
    fprintf('Época %2d/%2d  ->  accuracy Mes0(val) [exploit] = %.4f\n', ep, maxEpochs, valAcc);

    if valAcc > bestValAcc
        bestValAcc = valAcc;
        bestA_inv = A_inv;
        bestB = b;
        epochsSinceImprovement = 0;
    else
        epochsSinceImprovement = epochsSinceImprovement + 1;
        if epochsSinceImprovement >= patience
            fprintf('Sin mejora en %d épocas seguidas, deteniendo (early stopping).\n', patience);
            break;
        end
    end
end

%% CONGELADO: se restaura el mejor estado visto en validación y no se vuelve a tocar
A_inv = bestA_inv;
b = bestB;
fprintf('\nBandit congelado. Mejor accuracy Mes0(val) durante warm-start: %.4f\n', bestValAcc);

%% EVALUACIÓN: Mes0(validation, baseline) + Mes1..MesN (solo explotación, sin más aprendizaje)
% N = Shared.NUM_VALID_TEST_MONTHS: solo meses con anotacion consistente con Mes0.
numMeses = Shared.NUM_VALID_TEST_MONTHS;
accByMonth = nan(1, numMeses + 1);
f1ByMonth = nan(1, numMeses + 1);
allMonthLabels = ["Mes0(val)", "Mes1", "Mes2", "Mes3", "Mes4", "Mes5", "Mes6"];
monthLabels = allMonthLabels(1:numMeses + 1);

predVal = predictBanditExploit(A_inv, b, X_val, classNames);
[accByMonth(1), f1ByMonth(1)] = evaluateFold(y_val(:), predVal(:), classNames, 'Bandit - Mes0(val)');
fprintf('Bandit  Mes0(val): accuracy=%.4f  macroF1=%.4f\n', accByMonth(1), f1ByMonth(1));

for m = 1:numMeses
    predTest = predictBanditExploit(A_inv, b, X_test{m}, classNames);
    [accByMonth(m + 1), f1ByMonth(m + 1)] = evaluateFold(y_test{m}(:), predTest(:), classNames, sprintf('Bandit - Mes%d', m));
    fprintf('Bandit  Mes%d: accuracy=%.4f  macroF1=%.4f\n', m, accByMonth(m + 1), f1ByMonth(m + 1));
end

%% GUARDAR RESULTADOS
if ~exist('Models', 'dir')
    mkdir('Models');
end
resultsFile = sprintf('Models/Bandit_Results_%s_%s.mat', tamano, datestr(now, 'dd-mm-yyyy_HH-MM-ss'));
save(resultsFile, 'A_inv', 'b', 'alpha', 'accByMonth', 'f1ByMonth', 'monthLabels', 'classNames', 'classWeights', 'cacheFile', 'tamano');
fprintf('\nResultados guardados en: %s\n', resultsFile);

end


%% ======================================================
%% FUNCTION TO PREDICT WITH A FROZEN BANDIT (PURE EXPLOITATION)
%% ======================================================
function predLabels = predictBanditExploit(A_inv, b, X, classNames)
    % Sin el termino de incertidumbre UCB: ese termino solo tiene sentido
    % mientras se sigue aprendiendo. En congelado, evaluar con el
    % introduciria una regla de decision distinta a la que de hecho se
    % aprendio.
    numClasses = numel(classNames);
    theta = zeros(size(X, 1), numClasses);
    for i = 1:numClasses
        theta(:, i) = A_inv{i} * b{i};
    end
    scores = theta' * X; % numClasses x N
    [~, armIdx] = max(scores, [], 1);
    predLabels = categorical(classNames(armIdx), classNames);
end


%% ======================================================
%% FUNCTION TO EVALUATE ONE FOLD (accuracy, macro-F1, confusion matrix)
%% Identica a la usada en trainSoftmaxHead.m, para que ambos paradigmas se
%% midan exactamente con el mismo criterio.
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
