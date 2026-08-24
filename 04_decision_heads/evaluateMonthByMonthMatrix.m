% #################################################################
% Tarea 3 (plan de respuesta a revisores, 2026-07-22..24): matriz de
% evaluación mes x mes (7x7) para la Condición A (softmax). Generaliza el
% protocolo actual (solo "entrena en Mes0, evalúa en Mes1-6") a: para cada
% mes_train en {Mes0,...,Mes6}, se entrena una cabeza softmax nueva
% (backbone congelado, igual que trainSoftmaxHead.m) usando SOLO datos de
% ese mes, y se evalúa contra los 7 meses completos (incluida la diagonal,
% mes_train==mes_eval).
%
% Alcance confirmado con el usuario: SOLO condición A (softmax). Las
% condiciones B/C (LinUCB congelado/adaptativo) mantienen su protocolo
% temporal ya existente (warm-start en Mes0, adaptación prequential
% Mes1->Mes6) -- generalizar "iniciar el bandit en Mes3" no tiene la misma
% semántica metodológica que reentrenar una cabeza softmax, así que no se
% mezcla en esta matriz.
%
% Partición por mes de referencia: como el cache (buildFeatureCache.m)
% solo separa train/val para Mes0 (X_train/X_val) y guarda Mes1..Mes6
% como un solo bloque por mes (X_test{m}), aquí se hace un split 80/20
% ad-hoc dentro de cada mes de referencia (misma proporción que usa
% generateSpectrogramDataset.m para Mes0), SOLO para entrenar la cabeza de
% ese mes -- la evaluación (incluida la celda diagonal) siempre usa el mes
% COMPLETO, para que todas las celdas de una fila sean comparables entre
% sí (ninguna celda evalúa sobre un subconjunto recortado).
% #################################################################

clear; clc;
rng(9);
tamano = 'grande'; % esta matriz se corre sobre la variante grande (arquitectura principal)

%% CARGAR EL CACHE DE FEATURES
if strcmp(tamano, 'grande')
    files = dir(fullfile('Models', 'FrozenFeatures_*.mat'));
    esOtraVariante = contains({files.name}, {'_ann_', '_mediano_', '_pequeno_'});
    files = files(~esOtraVariante);
else
    files = dir(fullfile('Models', sprintf('FrozenFeatures_%s_*.mat', tamano)));
end
if isempty(files)
    error('No se encontró ningún cache de features para tamano=%s', tamano);
end
[~, idx] = max([files.datenum]);
cacheFile = fullfile(files(idx).folder, files(idx).name);
fprintf('Cache usado: %s\n', cacheFile);
load(cacheFile, 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', 'classNames');

numClasses = numel(classNames);
d_model = size(X_train, 1);
numMesesTest = numel(X_test); % Mes1..Mes6

%% ENSAMBLAR "TODOS LOS MESES" EN UNA SOLA ESTRUCTURA (Mes0 = train+val combinados; Mes1..6 = X_test)
% Para Mes0 se combina X_train+X_val en un solo bloque "Mes0 completo", de
% la misma forma que Mes1..6 son un solo bloque por mes -- así el split
% 80/20 ad-hoc de esta matriz se aplica de forma consistente a los 7
% meses, y la evaluación diagonal de Mes0 usa el Mes0 completo, no solo
% X_val (que es un subconjunto ya usado para early stopping en el
% pipeline principal).
numMesesTotal = numMesesTest + 1; % Mes0..Mes6
X_porMes = cell(1, numMesesTotal);
y_porMes = cell(1, numMesesTotal);
X_porMes{1} = [X_train, X_val];
y_porMes{1} = [y_train(:); y_val(:)];
for m = 1:numMesesTest
    X_porMes{m + 1} = X_test{m};
    y_porMes{m + 1} = y_test{m}(:);
end
monthNames = arrayfun(@(m) sprintf('Mes%d', m), 0:numMesesTest, 'UniformOutput', false);

%% MATRIZ DE RESULTADOS: filas=mes de entrenamiento, columnas=mes de evaluación
accMatrix = nan(numMesesTotal, numMesesTotal);

for mTrain = 1:numMesesTotal
    fprintf('\n========================================================================\n');
    fprintf(' Entrenando cabeza softmax con %s como referencia (%d/%d)\n', monthNames{mTrain}, mTrain, numMesesTotal);
    fprintf('========================================================================\n');

    Xm = X_porMes{mTrain};
    ym = y_porMes{mTrain};
    Nm = size(Xm, 2);

    % Split 80/20 ad-hoc SOLO para entrenar+early-stopping de esta cabeza
    % (misma proporción que usa generateSpectrogramDataset.m para Mes0).
    rng(9); % misma semilla en cada mes, para que el split sea reproducible por separado
    idxPerm = randperm(Nm);
    nTrain = floor(Nm * 0.8);
    idxTrain = idxPerm(1:nTrain);
    idxVal = idxPerm(nTrain + 1:end);

    XTrainM = Xm(:, idxTrain);
    yTrainM = ym(idxTrain);
    XValM = Xm(:, idxVal);
    yValM = ym(idxVal);

    counts = countcats(categorical(yTrainM));
    classWeightsM = sum(counts) ./ (numClasses * counts);

    layers = [
        featureInputLayer(d_model, "Name", "context_input", "Normalization", "zscore")
        fullyConnectedLayer(numClasses, "Name", "fc_head")
        softmaxLayer("Name", "softmax_head")
        classificationLayer("Name", "classoutput_head", "Classes", classNames, "ClassWeights", classWeightsM)];

    options = trainingOptions('adam', ...
        'InitialLearnRate', 0.01, ...
        'L2Regularization', 0.001, ...
        'MaxEpochs', 200, ...
        'MiniBatchSize', 128, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 0, ...
        'ValidationData', {XValM', yValM(:)}, ...
        'ValidationFrequency', max(1, floor(nTrain / 128)), ...
        'ValidationPatience', 10, ...
        'Plots', 'none');

    netM = trainNetwork(XTrainM', yTrainM(:), layers, options);

    % EVALUAR contra los 7 meses COMPLETOS (incluida la diagonal)
    for mEval = 1:numMesesTotal
        XEval = X_porMes{mEval};
        yEval = y_porMes{mEval};
        predEval = classify(netM, XEval');
        accMatrix(mTrain, mEval) = mean(predEval(:) == yEval(:));
        fprintf('  Evaluado en %s: accuracy=%.4f\n', monthNames{mEval}, accMatrix(mTrain, mEval));
    end
end

%% GUARDAR RESULTADOS
if ~exist('Models', 'dir')
    mkdir('Models');
end
resultsFile = sprintf('Models/MonthByMonthMatrix_%s_%s.mat', tamano, datestr(now, 'dd-mm-yyyy_HH-MM-ss'));
save(resultsFile, 'accMatrix', 'monthNames', 'classNames', 'tamano', 'cacheFile');
fprintf('\nMatriz de resultados guardada en: %s\n', resultsFile);

%% VALIDACIÓN: la diagonal debe ser razonablemente alta (cada cabeza se evalúa en su propio mes)
fprintf('\n--- Validación: accuracy en la diagonal (mismo mes de train y eval) ---\n');
for m = 1:numMesesTotal
    fprintf('%s: %.4f\n', monthNames{m}, accMatrix(m, m));
end

%% FIGURA: heatmap 7x7
outDir = 'FigurasReales';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

fig = figure('Visible', 'off', 'Position', [100 100 800 700]);
h = heatmap(monthNames, monthNames, accMatrix * 100, 'Colormap', parula, 'ColorLimits', [50 100]);
h.Title = 'Matriz de generalización cruzada entre sesiones (Condición A, softmax)';
h.XLabel = 'Mes de EVALUACIÓN';
h.YLabel = 'Mes de ENTRENAMIENTO';
exportgraphics(fig, fullfile(outDir, 'matriz_mes_x_mes_condicionA.png'), 'Resolution', 200);
close(fig);

fprintf('\nFigura guardada en: %s\n', fullfile(pwd, outDir, 'matriz_mes_x_mes_condicionA.png'));
