% #################################################################
% Validacion adicional (pedido explicito 2026-08-07), analoga a
% trainKNNVariants.m pero para SVM: 2 variantes con hiperparámetros FIJOS
% (no elegidos por accuracy en Mes0-val), sobre el mismo cache limpio.
%
% Variante 2: kernel polinomial de grado 2, BoxConstraint por defecto (1).
% Variante 3: kernel lineal con BoxConstraint mucho más alto (10, menos
%             regularización) -- para ver si un SVM lineal, con otra
%             configuración de regularización, muestra el mismo patrón.
% #################################################################

clear; clc;
rng(9);
tamano = 'ann';

cacheFiles = dir(fullfile('Models', sprintf('FrozenFeatures_%s_*.mat', tamano)));
if isempty(cacheFiles)
    error('No se encontró Models/FrozenFeatures_%s_*.mat.', tamano);
end
[~, newestIdx] = max([cacheFiles.datenum]);
cacheFile = fullfile(cacheFiles(newestIdx).folder, cacheFiles(newestIdx).name);
fprintf('Cache de features usado (SVM variantes, %s): %s\n', tamano, cacheFile);
load(cacheFile, 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', 'classNames');
classNames = categorical(cellstr(classNames));

numClasses = numel(classNames);

counts = countcats(y_train(:));
classWeights = sum(counts) ./ (numClasses * counts);
priorWeights = classWeights ./ sum(classWeights);

muTrain = mean(X_train, 2);
sigmaTrain = std(X_train, 0, 2);
sigmaTrain(sigmaTrain == 0) = 1;
normalize = @(X) (X - muTrain) ./ sigmaTrain;
X_train_n = normalize(X_train);
X_val_n = normalize(X_val);

% Mismo submuestreo que trainSVMHead.m, para mantener costo computacional
% comparable (SVM multiclase es O(n^2)-O(n^3)).
maxPerClass = 1500;
rng(9);
keepIdx = [];
for c = 1:numClasses
    classIdx = find(y_train(:) == classNames(c));
    if numel(classIdx) > maxPerClass
        classIdx = classIdx(randperm(numel(classIdx), maxPerClass));
    end
    keepIdx = [keepIdx; classIdx];
end
X_train_fit = X_train_n(:, keepIdx);
y_train_fit = y_train(keepIdx);
fprintf('Muestras usadas para entrenar (tras submuestreo): %d\n', numel(y_train_fit));

variantes = struct( ...
    'nombre', {'svm_poly2_C1', 'svm_linear_C10'}, ...
    'kernel', {'polynomial', 'linear'}, ...
    'polyOrder', {2, []}, ...
    'boxConstraint', {1, 10} ...
);

numMeses = Shared.NUM_VALID_TEST_MONTHS;
allMonthLabels = ["Mes0(val)", "Mes1", "Mes2", "Mes3", "Mes4", "Mes5", "Mes6"];
monthLabels = allMonthLabels(1:numMeses + 1);

for v = 1:numel(variantes)
    var = variantes(v);
    fprintf('\n=== Variante: %s (kernel=%s, C=%g) ===\n', var.nombre, var.kernel, var.boxConstraint);

    if strcmp(var.kernel, 'polynomial')
        t = templateSVM('KernelFunction', 'polynomial', 'PolynomialOrder', var.polyOrder, ...
            'BoxConstraint', var.boxConstraint, 'Standardize', false);
    else
        t = templateSVM('KernelFunction', 'linear', ...
            'BoxConstraint', var.boxConstraint, 'Standardize', false);
    end
    svmMdl = fitcecoc(X_train_fit', y_train_fit(:), 'Learners', t, ...
        'ClassNames', classNames, 'Prior', priorWeights);

    accByMonth = nan(1, numMeses + 1);
    f1ByMonth = nan(1, numMeses + 1);

    predVal = predict(svmMdl, X_val_n');
    [accByMonth(1), f1ByMonth(1)] = evaluateFold(y_val(:), predVal, classNames);
    fprintf('  Mes0(val): accuracy=%.4f  macroF1=%.4f\n', accByMonth(1), f1ByMonth(1));

    for m = 1:numMeses
        X_test_n = normalize(X_test{m});
        predTest = predict(svmMdl, X_test_n');
        [accByMonth(m + 1), f1ByMonth(m + 1)] = evaluateFold(y_test{m}(:), predTest, classNames);
        fprintf('  Mes%d: accuracy=%.4f  macroF1=%.4f\n', m, accByMonth(m + 1), f1ByMonth(m + 1));
    end

    if ~exist('Models', 'dir')
        mkdir('Models');
    end
    resultsFile = sprintf('Models/SoftmaxHead_Results_%s_%s.mat', var.nombre, datestr(now, 'dd-mm-yyyy_HH-MM-ss'));
    save(resultsFile, 'svmMdl', 'accByMonth', 'f1ByMonth', 'monthLabels', 'classNames', 'classWeights', 'cacheFile', 'muTrain', 'sigmaTrain', 'var');
    fprintf('  Resultados guardados en: %s\n', resultsFile);
end

fprintf('\nVariantes de SVM completadas.\n');

function [acc, macroF1] = evaluateFold(trueLabels, predLabels, classNames)
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
