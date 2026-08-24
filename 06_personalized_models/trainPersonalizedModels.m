% #################################################################
% Modelos personalizados (pedido explicito 2026-08-11): en vez de un solo
% modelo entrenado con los 19 usuarios mezclados, entrenar 19 modelos
% INDEPENDIENTES por arquitectura -- cada uno usando SOLO los datos de un
% usuario -- y evaluar la curva Mes0(val)->Mes6 de cada persona por
% separado. Objetivo: aislar el drift temporal real de cada individuo,
% sin la variabilidad entre sujetos que promedia el modelo poblacional.
%
% Arquitecturas elegidas (las 2 configuraciones mas ESTABLES encontradas
% en el barrido de variantes de 2026-08-07/11, menor escalon Mes0->Mes1):
%   - SVM lineal, C=10           (caida poblacional 3.0%)
%   - kNN k=9, coseno, ponderado por distancia (caida poblacional 9.7%)
%
% Metodologia identica al resto del pipeline salvo el filtro por usuario:
%   - Split 80/20 por repeticion dentro de Mes0 (ya viene asi en el cache,
%     no se resortea nada aqui -- se filtra el cache existente por
%     userID_train/userID_val/userID_test).
%   - Los 19 usuarios estan presentes en train+val+Mes1..6 (confirmado
%     2026-08-11), asi que no hace falta excluir a nadie.
%   - Normalizacion z-score con muTrain/sigmaTrain calculados por usuario
%     (no con los estadisticos poblacionales), porque cada modelo
%     personal es una unidad de entrenamiento independiente.
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
fprintf('Cache de features usado (modelos personalizados, %s): %s\n', tamano, cacheFile);
load(cacheFile, 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', ...
    'userID_train', 'userID_val', 'userID_test', 'classNames');
classNames = categorical(cellstr(classNames));
numClasses = numel(classNames);

allUsers = unique(userID_train);
numMeses = Shared.NUM_VALID_TEST_MONTHS;
allMonthLabels = ["Mes0(val)", "Mes1", "Mes2", "Mes3", "Mes4", "Mes5", "Mes6"];
monthLabels = allMonthLabels(1:numMeses + 1);

fprintf('Usuarios a procesar: %d\n', numel(allUsers));

resultadosSVM = struct();
resultadosKNN = struct();

for u = 1:numel(allUsers)
    user = allUsers(u);
    fprintf('\n========== Usuario %s (%d/%d) ==========\n', user, u, numel(allUsers));

    idxTrain = (userID_train == user);
    idxVal = (userID_val == user);
    Xu_train = X_train(:, idxTrain);
    yu_train = y_train(idxTrain);
    Xu_val = X_val(:, idxVal);
    yu_val = y_val(idxVal);

    fprintf('  Frames: train=%d  val=%d\n', numel(yu_train), numel(yu_val));

    % Pesos de clase derivados SOLO de este usuario (Mes0 training)
    counts = countcats(yu_train(:));
    counts(counts == 0) = 1; % evitar división por cero si a este usuario le falta alguna clase
    classWeights = sum(counts) ./ (numClasses * counts);
    priorWeights = classWeights ./ sum(classWeights);

    % Normalización z-score con estadísticos de ESTE usuario
    muTrain = mean(Xu_train, 2);
    sigmaTrain = std(Xu_train, 0, 2);
    sigmaTrain(sigmaTrain == 0) = 1;
    normalize = @(X) (X - muTrain) ./ sigmaTrain;
    Xu_train_n = normalize(Xu_train);
    Xu_val_n = normalize(Xu_val);

    %% --- SVM lineal, C=10 (misma config que la variante mas estable) ---
    t = templateSVM('KernelFunction', 'linear', 'BoxConstraint', 10, 'Standardize', false);
    svmMdl = fitcecoc(Xu_train_n', yu_train(:), 'Learners', t, ...
        'ClassNames', classNames, 'Prior', priorWeights);

    accSVM = nan(1, numMeses + 1);
    predVal = predict(svmMdl, Xu_val_n');
    accSVM(1) = mean(predVal == yu_val(:));
    fprintf('  SVM  Mes0(val): accuracy=%.4f\n', accSVM(1));
    for m = 1:numMeses
        idxTest = (userID_test{m} == user);
        Xu_test_n = normalize(X_test{m}(:, idxTest));
        yu_test = y_test{m}(idxTest);
        predTest = predict(svmMdl, Xu_test_n');
        accSVM(m + 1) = mean(predTest == yu_test(:));
        fprintf('  SVM  Mes%d: accuracy=%.4f  (n=%d)\n', m, accSVM(m + 1), numel(yu_test));
    end
    resultadosSVM.(user).accByMonth = accSVM;
    resultadosSVM.(user).nTrain = numel(yu_train);
    resultadosSVM.(user).nVal = numel(yu_val);

    %% --- kNN k=9, coseno, ponderado por distancia ---
    knnMdl = fitcknn(Xu_train_n', yu_train(:), 'NumNeighbors', 9, ...
        'Distance', 'cosine', 'DistanceWeight', 'inverse', ...
        'Prior', priorWeights, 'ClassNames', classNames);

    accKNN = nan(1, numMeses + 1);
    predVal = predict(knnMdl, Xu_val_n');
    accKNN(1) = mean(predVal == yu_val(:));
    fprintf('  kNN  Mes0(val): accuracy=%.4f\n', accKNN(1));
    for m = 1:numMeses
        idxTest = (userID_test{m} == user);
        Xu_test_n = normalize(X_test{m}(:, idxTest));
        yu_test = y_test{m}(idxTest);
        predTest = predict(knnMdl, Xu_test_n');
        accKNN(m + 1) = mean(predTest == yu_test(:));
        fprintf('  kNN  Mes%d: accuracy=%.4f  (n=%d)\n', m, accKNN(m + 1), numel(yu_test));
    end
    resultadosKNN.(user).accByMonth = accKNN;
    resultadosKNN.(user).nTrain = numel(yu_train);
    resultadosKNN.(user).nVal = numel(yu_val);
end

if ~exist('Models', 'dir')
    mkdir('Models');
end
resultsFile = sprintf('Models/PersonalizedModels_svm_linear_C10_%s.mat', datestr(now, 'dd-mm-yyyy_HH-MM-ss'));
save(resultsFile, 'resultadosSVM', 'allUsers', 'monthLabels', 'cacheFile');
fprintf('\nResultados SVM personalizados guardados en: %s\n', resultsFile);

resultsFile2 = sprintf('Models/PersonalizedModels_knn_k9cosine_%s.mat', datestr(now, 'dd-mm-yyyy_HH-MM-ss'));
save(resultsFile2, 'resultadosKNN', 'allUsers', 'monthLabels', 'cacheFile');
fprintf('Resultados kNN personalizados guardados en: %s\n', resultsFile2);

%% RESUMEN: tabla usuario x mes para ambos modelos
fprintf('\n\n=== RESUMEN SVM lineal C=10 (accuracy %%, por usuario) ===\n');
fprintf('%-10s', 'Usuario');
for m = 1:numel(monthLabels)
    fprintf('%10s', monthLabels{m});
end
fprintf('\n');
for u = 1:numel(allUsers)
    user = allUsers(u);
    fprintf('%-10s', user);
    fprintf('%10.1f', resultadosSVM.(user).accByMonth * 100);
    fprintf('\n');
end

fprintf('\n=== RESUMEN kNN k=9 coseno (accuracy %%, por usuario) ===\n');
fprintf('%-10s', 'Usuario');
for m = 1:numel(monthLabels)
    fprintf('%10s', monthLabels{m});
end
fprintf('\n');
for u = 1:numel(allUsers)
    user = allUsers(u);
    fprintf('%-10s', user);
    fprintf('%10.1f', resultadosKNN.(user).accByMonth * 100);
    fprintf('\n');
end

fprintf('\nModelos personalizados completados.\n');
