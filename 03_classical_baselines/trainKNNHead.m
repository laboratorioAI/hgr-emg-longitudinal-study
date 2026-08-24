% #################################################################
% Algoritmo tradicional adicional (pedido explicito 2026-07-27): kNN sobre
% el mismo cache de features clasicos (MAV/RMS/SD/energia) usado por
% trainANNHead.m (Experimento 2A). Objetivo: ver si con un clasificador
% completamente distinto (no red neuronal, sin descenso de gradiente, sin
% backbone congelado de ningun tipo) el patron Mes0(val) como pico se
% mantiene, o si en algun algoritmo tradicional Mes0 vuelve a ser el mejor
% mes (el patron "clasico" de drift que se busca).
%
% Referencia de diseno: Benalcazar et al. (2018, EUSIPCO) uso ANN-DTW
% sobre features estadisticos de EMG; Barona Lopez et al. (2020, Sensors)
% uso SVM sobre el mismo tipo de features. kNN se agrega aqui como tercer
% clasificador clasico de referencia directa en literatura EMG (Oskoei &
% Hu, 2007, IEEE Trans. Biomed. Eng., "Support Vector Machine-Based
% Classification Scheme for Myoelectric Control Applied to Upper Limb" --
% compara kNN, LDA y SVM sobre features EMG clasicos).
%
% Mismo cache (Models/FrozenFeatures_ann_*.mat), mismo protocolo de
% evaluacion (Mes0-val + Mes1..Mes6, sin reentrenar despues) que
% trainANNHead.m / trainSoftmaxHead.m, para que la comparacion sea justa.
% #################################################################

clear; clc;
rng(9);
tamano = 'ann';

%% CARGAR EL CACHE DE FEATURES CLASICOS (el mismo que usa la ANN sin CNN)
cacheFiles = dir(fullfile('Models', sprintf('FrozenFeatures_%s_*.mat', tamano)));
if isempty(cacheFiles)
    error('No se encontró Models/FrozenFeatures_%s_*.mat. Corre buildStatFeatureCache.m primero.', tamano);
end
[~, newestIdx] = max([cacheFiles.datenum]);
cacheFile = fullfile(cacheFiles(newestIdx).folder, cacheFiles(newestIdx).name);
fprintf('Cache de features usado (kNN, %s): %s\n', tamano, cacheFile);
load(cacheFile, 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', 'classNames');
classNames = categorical(cellstr(classNames)); % fitcknn debe devolver predicciones categorical,
                                                % del mismo tipo que y_val/y_test (si ClassNames
                                                % se pasa como string, predict() devuelve string
                                                % y confusionmat falla por tipos distintos)

numClasses = numel(classNames);
numFeatures = size(X_train, 1);
fprintf('Dimensión de entrada (features clásicos): %d\n', numFeatures);

%% PESOS DE CLASE (mismo criterio balanceado que el resto del pipeline)
counts = countcats(y_train(:));
classWeights = sum(counts) ./ (numClasses * counts);
priorWeights = classWeights ./ sum(classWeights); % fitcknn usa 'Prior', no 'ClassWeights'
fprintf('\nPesos de clase (kNN, derivados de Mes0 training):\n');
for i = 1:numClasses
    fprintf('  %-10s peso=%.4f\n', classNames(i), classWeights(i));
end

%% NORMALIZACIÓN (zscore con estadísticos de Mes0-train, aplicados a todos los splits)
% Igual que el featureInputLayer('Normalization','zscore') de trainANNHead.m,
% pero explícito porque fitcknn no normaliza internamente.
muTrain = mean(X_train, 2);
sigmaTrain = std(X_train, 0, 2);
sigmaTrain(sigmaTrain == 0) = 1; % evitar división por cero en features constantes

normalize = @(X) (X - muTrain) ./ sigmaTrain;
X_train_n = normalize(X_train);
X_val_n = normalize(X_val);

%% ENTRENAMIENTO: kNN CON K SELECCIONADO POR VALIDACIÓN (Mes0-val)
% Se prueba un rango pequeño de k impares (evita empates en votación) y se
% elige el que maximiza accuracy en Mes0-val, igual que ValidationPatience
% actúa como criterio de selección de modelo en las redes neuronales.
kCandidates = [1, 3, 5, 7, 9, 11, 15, 21];
bestK = kCandidates(1);
bestValAcc = -Inf;
fprintf('\nBuscando mejor k (kNN) por accuracy en Mes0-val...\n');
for k = kCandidates
    mdlTry = fitcknn(X_train_n', y_train(:), 'NumNeighbors', k, ...
        'Distance', 'euclidean', 'Prior', priorWeights, 'ClassNames', classNames);
    predValTry = predict(mdlTry, X_val_n');
    accTry = mean(predValTry == y_val(:));
    fprintf('  k=%2d  accuracy(val)=%.4f\n', k, accTry);
    if accTry > bestValAcc
        bestValAcc = accTry;
        bestK = k;
    end
end
fprintf('Mejor k seleccionado: %d (accuracy val=%.4f)\n', bestK, bestValAcc);

knnMdl = fitcknn(X_train_n', y_train(:), 'NumNeighbors', bestK, ...
    'Distance', 'euclidean', 'Prior', priorWeights, 'ClassNames', classNames);

%% CONGELADO: de aquí en adelante solo se usa predict(), nunca más se reentrena.

%% EVALUACIÓN: Mes0(validation, baseline) + Mes1..MesN
numMeses = Shared.NUM_VALID_TEST_MONTHS;
accByMonth = nan(1, numMeses + 1);
f1ByMonth = nan(1, numMeses + 1);
allMonthLabels = ["Mes0(val)", "Mes1", "Mes2", "Mes3", "Mes4", "Mes5", "Mes6"];
monthLabels = allMonthLabels(1:numMeses + 1);

predVal = predict(knnMdl, X_val_n');
[accByMonth(1), f1ByMonth(1)] = evaluateFold(y_val(:), predVal, classNames, 'kNN - Mes0(val)');
fprintf('kNN  Mes0(val): accuracy=%.4f  macroF1=%.4f\n', accByMonth(1), f1ByMonth(1));

for m = 1:numMeses
    X_test_n = normalize(X_test{m});
    predTest = predict(knnMdl, X_test_n');
    [accByMonth(m + 1), f1ByMonth(m + 1)] = evaluateFold(y_test{m}(:), predTest, classNames, sprintf('kNN - Mes%d', m));
    fprintf('kNN  Mes%d: accuracy=%.4f  macroF1=%.4f\n', m, accByMonth(m + 1), f1ByMonth(m + 1));
end

%% GUARDAR RESULTADOS (mismo formato que SoftmaxHead_Results, tamaño 'knn')
if ~exist('Models', 'dir')
    mkdir('Models');
end
tamanoTag = 'knn';
resultsFile = sprintf('Models/SoftmaxHead_Results_%s_%s.mat', tamanoTag, datestr(now, 'dd-mm-yyyy_HH-MM-ss'));
save(resultsFile, 'knnMdl', 'bestK', 'accByMonth', 'f1ByMonth', 'monthLabels', 'classNames', 'classWeights', 'cacheFile', 'muTrain', 'sigmaTrain');
fprintf('\nResultados guardados en: %s\n', resultsFile);


%% ======================================================
%% FUNCTION TO EVALUATE ONE FOLD (accuracy, macro-F1, confusion matrix)
%% Idéntica a la usada en trainSoftmaxHead.m / trainANNHead.m.
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

    figure('Name', ['Confusion Matrix - ' foldName], 'Visible', 'off');
    matrixChart = confusionchart(confMat, classesCat);
    matrixChart.ColumnSummary = 'column-normalized';
    matrixChart.RowSummary = 'row-normalized';
    matrixChart.Title = ['Hand gestures - ' foldName];
    sortClasses(matrixChart, classesCat);
end
