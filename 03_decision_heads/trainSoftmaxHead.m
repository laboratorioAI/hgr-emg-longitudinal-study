% #################################################################
% PASO 2b: PARADIGMA A - CABEZA SOFTMAX SOBRE EL BACKBONE CONGELADO
% Reentrena solo fc_final+softmax (como featureInputLayer -> FC -> softmax)
% sobre los features congelados de Mes0 (dropout_2, 128-d), calculados por
% buildFeatureCache.m. Se congela tras entrenar y se evalúa SIN más
% actualizaciones en Mes0(validation, baseline) y Mes1..Mes6, para una
% comparación justa contra el LinUCB de context_bandit.m (mismo cache de
% features, mismo criterio de pesos de clase, mismo protocolo de
% evaluación).
%
% PARAMETRIZADO POR TAMAÑO (Tarea 2 / Experimento 2B): 'grande' (default),
% 'mediano', 'pequeno' -- busca el cache de ESE tamaño específicamente.
% #################################################################

function trainSoftmaxHead(tamano)

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
d_model = size(X_train, 1);

%% PESOS DE CLASE (misma fórmula balanceada usada en el backbone)
counts = countcats(y_train(:));
classWeights = sum(counts) ./ (numClasses * counts);
fprintf('\nPesos de clase (softmax, derivados de Mes0 training):\n');
for i = 1:numClasses
    fprintf('  %-10s peso=%.4f\n', classNames(i), classWeights(i));
end

%% ARQUITECTURA: SOLO LA CABEZA (el backbone ya está congelado / no se toca)
layers = [
    featureInputLayer(d_model, "Name", "context_input", "Normalization", "zscore")
    fullyConnectedLayer(numClasses, "Name", "fc_head")
    softmaxLayer("Name", "softmax_head")
    classificationLayer("Name", "classoutput_head", "Classes", classNames, "ClassWeights", classWeights)];

% NOTA: la normalizacion zscore del featureInputLayer es importante para que
% el optimizador converja de forma fiable (verificado con datos sinteticos:
% sin ella, con LR=0.001, se queda estancado cerca del nivel de azar aunque
% las clases sean perfectamente separables).
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.01, ...
    'L2Regularization', 0.001, ...
    'MaxEpochs', 200, ...
    'MiniBatchSize', 128, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', 0, ...
    'ValidationData', {X_val', y_val(:)}, ...
    'ValidationFrequency', max(1, floor(size(X_train, 2) / 128)), ...
    'ValidationPatience', 10, ...
    'Plots', 'training-progress');

fprintf('\nEntrenando cabeza softmax sobre features congeladas de Mes0...\n');
softmaxNet = trainNetwork(X_train', y_train(:), layers, options);

%% CONGELADO: de aquí en adelante solo se usa classify(), nunca más se reentrena.

%% EVALUACIÓN: Mes0(validation, baseline) + Mes1..MesN
% N = Shared.NUM_VALID_TEST_MONTHS: solo se incluyen los meses con
% anotacion (groundTruthIndex) consistente con Mes0. Subir ese numero en
% Shared.m en cuanto los meses restantes esten anotados igual.
numMeses = Shared.NUM_VALID_TEST_MONTHS;
accByMonth = nan(1, numMeses + 1);   % 1 = baseline (Mes0 val), 2..(numMeses+1) = Mes1..MesN
f1ByMonth = nan(1, numMeses + 1);
allMonthLabels = ["Mes0(val)", "Mes1", "Mes2", "Mes3", "Mes4", "Mes5", "Mes6"];
monthLabels = allMonthLabels(1:numMeses + 1);

predVal = classify(softmaxNet, X_val');
[accByMonth(1), f1ByMonth(1)] = evaluateFold(y_val(:), predVal, classNames, 'Softmax - Mes0(val)');
fprintf('Softmax  Mes0(val): accuracy=%.4f  macroF1=%.4f\n', accByMonth(1), f1ByMonth(1));

for m = 1:numMeses
    predTest = classify(softmaxNet, X_test{m}');
    [accByMonth(m + 1), f1ByMonth(m + 1)] = evaluateFold(y_test{m}(:), predTest, classNames, sprintf('Softmax - Mes%d', m));
    fprintf('Softmax  Mes%d: accuracy=%.4f  macroF1=%.4f\n', m, accByMonth(m + 1), f1ByMonth(m + 1));
end

%% GUARDAR RESULTADOS
if ~exist('Models', 'dir')
    mkdir('Models');
end
resultsFile = sprintf('Models/SoftmaxHead_Results_%s_%s.mat', tamano, datestr(now, 'dd-mm-yyyy_HH-MM-ss'));
save(resultsFile, 'softmaxNet', 'accByMonth', 'f1ByMonth', 'monthLabels', 'classNames', 'classWeights', 'cacheFile', 'tamano');
fprintf('\nResultados guardados en: %s\n', resultsFile);

end


%% ======================================================
%% FUNCTION TO EVALUATE ONE FOLD (accuracy, macro-F1, confusion matrix)
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
