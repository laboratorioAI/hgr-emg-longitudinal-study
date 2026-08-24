% #################################################################
% Experimento 2A (Tarea 2 del plan de respuesta a revisores, 2026-07-23):
% RED SIN CNN -- reemplaza trainSoftmaxHead.m para el caso 'ann': en vez
% de una sola capa (FC+softmax) sobre el cache de features clásicos, usa
% una red densa pequeña de 2 capas ocultas (64->32->numClasses), fiel al
% diseño acordado para "red sin CNN" (features estadísticos + ANN
% pequeña, patrón de Benalcázar et al. 2018 / Barona López et al. 2020).
%
% Comparte el mismo cache (Models/FrozenFeatures_ann_*.mat, generado por
% buildStatFeatureCache.m), el mismo protocolo de evaluación
% (Mes0-val + Mes1..Mes6, sin reentrenar después) y el mismo criterio de
% pesos de clase que trainSoftmaxHead.m -- así la comparación entre
% "Condición A con backbone CNN+Transformer" y "Condición A con red ANN
% sin CNN" es igual de justa que la comparación entre condiciones A/B/C.
% #################################################################

clear; clc;
rng(9);
tamano = 'ann';

%% CARGAR EL CACHE DE FEATURES CLÁSICOS
cacheFiles = dir(fullfile('Models', sprintf('FrozenFeatures_%s_*.mat', tamano)));
if isempty(cacheFiles)
    error('No se encontró Models/FrozenFeatures_%s_*.mat. Corre buildStatFeatureCache.m primero.', tamano);
end
[~, newestIdx] = max([cacheFiles.datenum]);
cacheFile = fullfile(cacheFiles(newestIdx).folder, cacheFiles(newestIdx).name);
fprintf('Cache de features usado (%s): %s\n', tamano, cacheFile);
load(cacheFile, 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', 'classNames');

numClasses = numel(classNames);
numFeatures = size(X_train, 1);
fprintf('Dimensión de entrada (features clásicos): %d\n', numFeatures);

%% PESOS DE CLASE (misma fórmula que trainSoftmaxHead.m / backbone)
counts = countcats(y_train(:));
classWeights = sum(counts) ./ (numClasses * counts);
fprintf('\nPesos de clase (ANN sin CNN, derivados de Mes0 training):\n');
for i = 1:numClasses
    fprintf('  %-10s peso=%.4f\n', classNames(i), classWeights(i));
end

%% ARQUITECTURA: RED DENSA PEQUEÑA, SIN NINGUNA CAPA CONVOLUCIONAL NI DE ATENCIÓN
% 2 capas ocultas (64 -> 32), tamaño deliberadamente pequeño: el objetivo
% de este experimento es la ausencia de sesgo convolucional, no competir
% en capacidad con el backbone CNN+Transformer.
layers = [
    featureInputLayer(numFeatures, "Name", "input", "Normalization", "zscore")
    fullyConnectedLayer(64, "Name", "fc1")
    reluLayer("Name", "relu1")
    dropoutLayer(0.3, "Name", "dropout1")
    fullyConnectedLayer(32, "Name", "fc2")
    reluLayer("Name", "relu2")
    fullyConnectedLayer(numClasses, "Name", "fc_head")
    softmaxLayer("Name", "softmax_head")
    classificationLayer("Name", "classoutput_head", "Classes", classNames, "ClassWeights", classWeights)];

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'L2Regularization', 0.001, ...
    'MaxEpochs', 60, ...
    'MiniBatchSize', 128, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', 0, ...
    'ValidationData', {X_val', y_val(:)}, ...
    'ValidationFrequency', max(1, floor(size(X_train, 2) / 128)), ...
    'ValidationPatience', 8, ...
    'Plots', 'training-progress');

fprintf('\nEntrenando ANN sin CNN (red densa 64->32) sobre features clásicos de Mes0...\n');
annNet = trainNetwork(X_train', y_train(:), layers, options);

%% CONGELADO: de aquí en adelante solo se usa classify(), nunca más se reentrena.

%% EVALUACIÓN: Mes0(validation, baseline) + Mes1..MesN
numMeses = Shared.NUM_VALID_TEST_MONTHS;
accByMonth = nan(1, numMeses + 1);
f1ByMonth = nan(1, numMeses + 1);
allMonthLabels = ["Mes0(val)", "Mes1", "Mes2", "Mes3", "Mes4", "Mes5", "Mes6"];
monthLabels = allMonthLabels(1:numMeses + 1);

predVal = classify(annNet, X_val');
[accByMonth(1), f1ByMonth(1)] = evaluateFold(y_val(:), predVal, classNames, 'ANN-sin-CNN - Mes0(val)');
fprintf('ANN-sin-CNN  Mes0(val): accuracy=%.4f  macroF1=%.4f\n', accByMonth(1), f1ByMonth(1));

for m = 1:numMeses
    predTest = classify(annNet, X_test{m}');
    [accByMonth(m + 1), f1ByMonth(m + 1)] = evaluateFold(y_test{m}(:), predTest, classNames, sprintf('ANN-sin-CNN - Mes%d', m));
    fprintf('ANN-sin-CNN  Mes%d: accuracy=%.4f  macroF1=%.4f\n', m, accByMonth(m + 1), f1ByMonth(m + 1));
end

%% GUARDAR RESULTADOS (mismo formato que SoftmaxHead_Results, para que compareParadigms.m y
%% los scripts de figuras puedan tratarlo igual, apuntando al tamaño 'ann')
if ~exist('Models', 'dir')
    mkdir('Models');
end
resultsFile = sprintf('Models/SoftmaxHead_Results_%s_%s.mat', tamano, datestr(now, 'dd-mm-yyyy_HH-MM-ss'));
softmaxNet = annNet; %#ok<NASGU> % alias para compatibilidad con el nombre de variable que usan los demás scripts
save(resultsFile, 'annNet', 'softmaxNet', 'accByMonth', 'f1ByMonth', 'monthLabels', 'classNames', 'classWeights', 'cacheFile', 'tamano');
fprintf('\nResultados guardados en: %s\n', resultsFile);


%% ======================================================
%% FUNCTION TO EVALUATE ONE FOLD (accuracy, macro-F1, confusion matrix)
%% Idéntica a la usada en trainSoftmaxHead.m / context_bandit.m.
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
