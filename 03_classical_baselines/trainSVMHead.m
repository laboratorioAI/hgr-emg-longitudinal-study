% #################################################################
% Algoritmo tradicional adicional (pedido explicito 2026-07-27): SVM
% multiclase (fitcecoc, uno-contra-uno de SVMs binarios) sobre el mismo
% cache de features clasicos (MAV/RMS/SD/energia) usado por
% trainANNHead.m / trainKNNHead.m (Experimento 2A).
%
% Referencia directa: Barona Lopez et al. (2020, Sensors) uso SVM sobre
% este mismo tipo de features EMG estadisticos; Oskoei & Hu (2007, IEEE
% Trans. Biomed. Eng.) es la referencia clasica de SVM para control
% mioelectrico. Kernel elegido por validacion (Mes0-val) entre lineal y
% RBF, igual criterio de seleccion de modelo que k en trainKNNHead.m.
%
% Mismo cache (Models/FrozenFeatures_ann_*.mat), mismo protocolo de
% evaluacion (Mes0-val + Mes1..Mes6, sin reentrenar despues).
% #################################################################

clear; clc;
rng(9);
tamano = 'ann';

%% CARGAR EL CACHE DE FEATURES CLASICOS
cacheFiles = dir(fullfile('Models', sprintf('FrozenFeatures_%s_*.mat', tamano)));
if isempty(cacheFiles)
    error('No se encontró Models/FrozenFeatures_%s_*.mat. Corre buildStatFeatureCache.m primero.', tamano);
end
[~, newestIdx] = max([cacheFiles.datenum]);
cacheFile = fullfile(cacheFiles(newestIdx).folder, cacheFiles(newestIdx).name);
fprintf('Cache de features usado (SVM, %s): %s\n', tamano, cacheFile);
load(cacheFile, 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', 'classNames');
classNames = categorical(cellstr(classNames)); % fitcecoc debe devolver predicciones categorical,
                                                % del mismo tipo que y_val/y_test (si ClassNames
                                                % se pasa como string, predict() devuelve string
                                                % y confusionmat falla por tipos distintos)

numClasses = numel(classNames);
numFeatures = size(X_train, 1);
fprintf('Dimensión de entrada (features clásicos): %d\n', numFeatures);

%% PESOS DE CLASE (mismo criterio balanceado que el resto del pipeline)
counts = countcats(y_train(:));
classWeights = sum(counts) ./ (numClasses * counts);
priorWeights = classWeights ./ sum(classWeights);
fprintf('\nPesos de clase (SVM, derivados de Mes0 training):\n');
for i = 1:numClasses
    fprintf('  %-10s peso=%.4f\n', classNames(i), classWeights(i));
end

%% NORMALIZACIÓN (zscore con estadísticos de Mes0-train)
muTrain = mean(X_train, 2);
sigmaTrain = std(X_train, 0, 2);
sigmaTrain(sigmaTrain == 0) = 1;

normalize = @(X) (X - muTrain) ./ sigmaTrain;
X_train_n = normalize(X_train);
X_val_n = normalize(X_val);

%% SUBMUESTREO PARA ENTRENAMIENTO (SVM multiclase es O(n^2)-O(n^3) por par;
%% con miles de frames por época y 8 épocas equivalentes ya cubiertas por
%% las otras cabezas, se usa una muestra estratificada por clase para
%% mantener el tiempo de entrenamiento razonable, igual que cualquier SVM
%% aplicado a datasets EMG de este tamaño en la literatura citada arriba).
maxPerClass = 1500;
rng(9);
keepIdx = [];
for c = 1:numClasses
    classIdx = find(y_train(:) == classNames(c));
    if numel(classIdx) > maxPerClass
        classIdx = classIdx(randperm(numel(classIdx), maxPerClass));
    end
    keepIdx = [keepIdx; classIdx]; %#ok<AGROW>
end
X_train_fit = X_train_n(:, keepIdx);
y_train_fit = y_train(keepIdx);
fprintf('Muestras usadas para entrenar SVM (tras submuestreo): %d\n', numel(y_train_fit));

%% ENTRENAMIENTO: PROBAR KERNEL LINEAL Y RBF, ELEGIR POR ACCURACY EN Mes0-val
kernels = {'linear', 'gaussian'};
bestKernel = kernels{1};
bestValAcc = -Inf;
fprintf('\nBuscando mejor kernel (SVM) por accuracy en Mes0-val...\n');
for i = 1:numel(kernels)
    kernel = kernels{i};
    t = templateSVM('KernelFunction', kernel, 'Standardize', false);
    mdlTry = fitcecoc(X_train_fit', y_train_fit(:), 'Learners', t, ...
        'ClassNames', classNames, 'Prior', priorWeights);
    predValTry = predict(mdlTry, X_val_n');
    accTry = mean(predValTry == y_val(:));
    fprintf('  kernel=%-10s accuracy(val)=%.4f\n', kernel, accTry);
    if accTry > bestValAcc
        bestValAcc = accTry;
        bestKernel = kernel;
    end
end
fprintf('Mejor kernel seleccionado: %s (accuracy val=%.4f)\n', bestKernel, bestValAcc);

t = templateSVM('KernelFunction', bestKernel, 'Standardize', false);
svmMdl = fitcecoc(X_train_fit', y_train_fit(:), 'Learners', t, ...
    'ClassNames', classNames, 'Prior', priorWeights);

%% CONGELADO: de aquí en adelante solo se usa predict(), nunca más se reentrena.

%% EVALUACIÓN: Mes0(validation, baseline) + Mes1..MesN
numMeses = Shared.NUM_VALID_TEST_MONTHS;
accByMonth = nan(1, numMeses + 1);
f1ByMonth = nan(1, numMeses + 1);
allMonthLabels = ["Mes0(val)", "Mes1", "Mes2", "Mes3", "Mes4", "Mes5", "Mes6"];
monthLabels = allMonthLabels(1:numMeses + 1);

predVal = predict(svmMdl, X_val_n');
[accByMonth(1), f1ByMonth(1)] = evaluateFold(y_val(:), predVal, classNames, 'SVM - Mes0(val)');
fprintf('SVM  Mes0(val): accuracy=%.4f  macroF1=%.4f\n', accByMonth(1), f1ByMonth(1));

for m = 1:numMeses
    X_test_n = normalize(X_test{m});
    predTest = predict(svmMdl, X_test_n');
    [accByMonth(m + 1), f1ByMonth(m + 1)] = evaluateFold(y_test{m}(:), predTest, classNames, sprintf('SVM - Mes%d', m));
    fprintf('SVM  Mes%d: accuracy=%.4f  macroF1=%.4f\n', m, accByMonth(m + 1), f1ByMonth(m + 1));
end

%% GUARDAR RESULTADOS (mismo formato que SoftmaxHead_Results, tamaño 'svm')
if ~exist('Models', 'dir')
    mkdir('Models');
end
tamanoTag = 'svm';
resultsFile = sprintf('Models/SoftmaxHead_Results_%s_%s.mat', tamanoTag, datestr(now, 'dd-mm-yyyy_HH-MM-ss'));
save(resultsFile, 'svmMdl', 'bestKernel', 'accByMonth', 'f1ByMonth', 'monthLabels', 'classNames', 'classWeights', 'cacheFile', 'muTrain', 'sigmaTrain');
fprintf('\nResultados guardados en: %s\n', resultsFile);


%% ======================================================
%% FUNCTION TO EVALUATE ONE FOLD (accuracy, macro-F1, confusion matrix)
%% Idéntica a la usada en trainSoftmaxHead.m / trainANNHead.m / trainKNNHead.m.
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
