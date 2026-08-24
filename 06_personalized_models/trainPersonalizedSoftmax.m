% #################################################################
% Extension del experimento de modelos personalizados (2026-08-11) a un
% backbone CNN-Transformer-encoder, para confirmar si el patron de gran
% fluctuacion inter-usuario (encontrado en SVM/kNN personalizados) tambien
% aparece con la representacion APRENDIDA de 32-d (backbone 'pequeno',
% d_model=32), no solo con features estadisticos hechos a mano.
%
% Backbone elegido: 'pequeno' (mas barato de re-evaluar 19 veces que
% 'grande'; el backbone en si NO se reentrena por usuario -- ya esta
% congelado en dropout_2 -- solo se reentrena la cabeza Softmax, igual
% que trainSoftmaxHead.m pero filtrando por userID).
% #################################################################

clear; clc;
rng(9);
tamano = 'pequeno';

cacheFiles = dir(fullfile('Models', sprintf('FrozenFeatures_%s_*.mat', tamano)));
[~, newestIdx] = max([cacheFiles.datenum]);
cacheFile = fullfile(cacheFiles(newestIdx).folder, cacheFiles(newestIdx).name);
fprintf('Cache de features usado (softmax personalizado, %s): %s\n', tamano, cacheFile);
load(cacheFile, 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', ...
    'userID_train', 'userID_val', 'userID_test', 'classNames');

numClasses = numel(classNames);
d_model = size(X_train, 1);
allUsers = unique(userID_train);
numMeses = Shared.NUM_VALID_TEST_MONTHS;
allMonthLabels = ["Mes0(val)", "Mes1", "Mes2", "Mes3", "Mes4", "Mes5", "Mes6"];
monthLabels = allMonthLabels(1:numMeses + 1);

fprintf('Usuarios a procesar: %d\n', numel(allUsers));

resultadosSoftmax = struct();

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

    counts = countcats(yu_train(:));
    counts(counts == 0) = 1;
    classWeights = sum(counts) ./ (numClasses * counts);

    layers = [
        featureInputLayer(d_model, "Name", "context_input", "Normalization", "zscore")
        fullyConnectedLayer(numClasses, "Name", "fc_head")
        softmaxLayer("Name", "softmax_head")
        classificationLayer("Name", "classoutput_head", "Classes", classNames, "ClassWeights", classWeights)];

    options = trainingOptions('adam', ...
        'InitialLearnRate', 0.01, ...
        'L2Regularization', 0.001, ...
        'MaxEpochs', 200, ...
        'MiniBatchSize', 64, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 0, ...
        'ValidationData', {Xu_val', yu_val(:)}, ...
        'ValidationFrequency', max(1, floor(size(Xu_train, 2) / 64)), ...
        'ValidationPatience', 10, ...
        'Plots', 'none');

    softmaxNet = trainNetwork(Xu_train', yu_train(:), layers, options);

    accSoftmax = nan(1, numMeses + 1);
    predVal = classify(softmaxNet, Xu_val');
    accSoftmax(1) = mean(predVal == yu_val(:));
    fprintf('  Softmax  Mes0(val): accuracy=%.4f\n', accSoftmax(1));
    for m = 1:numMeses
        idxTest = (userID_test{m} == user);
        Xu_test = X_test{m}(:, idxTest);
        yu_test = y_test{m}(idxTest);
        predTest = classify(softmaxNet, Xu_test');
        accSoftmax(m + 1) = mean(predTest == yu_test(:));
        fprintf('  Softmax  Mes%d: accuracy=%.4f  (n=%d)\n', m, accSoftmax(m + 1), numel(yu_test));
    end
    resultadosSoftmax.(user).accByMonth = accSoftmax;
    resultadosSoftmax.(user).nTrain = numel(yu_train);
    resultadosSoftmax.(user).nVal = numel(yu_val);
end

if ~exist('Models', 'dir')
    mkdir('Models');
end
resultsFile = sprintf('Models/PersonalizedModels_softmax_%s_%s.mat', tamano, datestr(now, 'dd-mm-yyyy_HH-MM-ss'));
save(resultsFile, 'resultadosSoftmax', 'allUsers', 'monthLabels', 'cacheFile', 'tamano');
fprintf('\nResultados Softmax personalizados guardados en: %s\n', resultsFile);

fprintf('\n\n=== RESUMEN Softmax backbone %s (accuracy %%, por usuario) ===\n', tamano);
fprintf('%-10s', 'Usuario');
for m = 1:numel(monthLabels)
    fprintf('%10s', monthLabels{m});
end
fprintf('\n');
for u = 1:numel(allUsers)
    user = allUsers(u);
    fprintf('%-10s', user);
    fprintf('%10.1f', resultadosSoftmax.(user).accByMonth * 100);
    fprintf('\n');
end

fprintf('\nSoftmax personalizado completado.\n');
