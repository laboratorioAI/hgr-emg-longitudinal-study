% #################################################################
% Validacion adicional (pedido explicito 2026-08-07): el usuario preguntó
% por qué el escalón Mes0(val)->Mes1 aparece y luego se mantiene plano en
% Mes1-6, en los 7 modelos ya entrenados (incluidos los que nunca tuvieron
% el bug de leakage de DatastoresStatFeatures). Para verificar que esto no
% es un artefacto de UN kNN/SVM en particular, se entrenan aquí 2
% variantes adicionales de kNN, con hiperparámetros FIJOS (no elegidos por
% accuracy en Mes0-val, para no encadenar el mismo criterio de selección
% que ya se identificó como una causa estructural del escalón), sobre el
% mismo cache limpio (post-fix del 2026-08-06).
%
% Variante 2: k=5 fijo, distancia euclidiana, sin peso por distancia
%             (mismo tipo que la variante principal pero sin el efecto de
%             "elegir el mejor k" -- k=5 es un valor razonable intermedio,
%             no seleccionado por rendimiento).
% Variante 3: k=9 fijo, distancia coseno (en vez de euclidiana), con voto
%             ponderado por distancia inversa -- cambia tanto la métrica
%             de distancia como el esquema de voto, para estresar si el
%             patrón depende de la métrica.
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
fprintf('Cache de features usado (kNN variantes, %s): %s\n', tamano, cacheFile);
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

variantes = struct( ...
    'nombre', {'knn_k5_euclidean_uniform', 'knn_k9_cosine_distweight'}, ...
    'k', {5, 9}, ...
    'distance', {'euclidean', 'cosine'}, ...
    'distWeight', {'equal', 'inverse'} ...
);

numMeses = Shared.NUM_VALID_TEST_MONTHS;
allMonthLabels = ["Mes0(val)", "Mes1", "Mes2", "Mes3", "Mes4", "Mes5", "Mes6"];
monthLabels = allMonthLabels(1:numMeses + 1);

for v = 1:numel(variantes)
    var = variantes(v);
    fprintf('\n=== Variante: %s (k=%d, distance=%s, weight=%s) ===\n', ...
        var.nombre, var.k, var.distance, var.distWeight);

    knnMdl = fitcknn(X_train_n', y_train(:), 'NumNeighbors', var.k, ...
        'Distance', var.distance, 'DistanceWeight', var.distWeight, ...
        'Prior', priorWeights, 'ClassNames', classNames);

    accByMonth = nan(1, numMeses + 1);
    f1ByMonth = nan(1, numMeses + 1);

    predVal = predict(knnMdl, X_val_n');
    [accByMonth(1), f1ByMonth(1)] = evaluateFold(y_val(:), predVal, classNames);
    fprintf('  Mes0(val): accuracy=%.4f  macroF1=%.4f\n', accByMonth(1), f1ByMonth(1));

    for m = 1:numMeses
        X_test_n = normalize(X_test{m});
        predTest = predict(knnMdl, X_test_n');
        [accByMonth(m + 1), f1ByMonth(m + 1)] = evaluateFold(y_test{m}(:), predTest, classNames);
        fprintf('  Mes%d: accuracy=%.4f  macroF1=%.4f\n', m, accByMonth(m + 1), f1ByMonth(m + 1));
    end

    if ~exist('Models', 'dir')
        mkdir('Models');
    end
    resultsFile = sprintf('Models/SoftmaxHead_Results_%s_%s.mat', var.nombre, datestr(now, 'dd-mm-yyyy_HH-MM-ss'));
    save(resultsFile, 'knnMdl', 'accByMonth', 'f1ByMonth', 'monthLabels', 'classNames', 'classWeights', 'cacheFile', 'muTrain', 'sigmaTrain', 'var');
    fprintf('  Resultados guardados en: %s\n', resultsFile);
end

fprintf('\nVariantes de kNN completadas.\n');

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
