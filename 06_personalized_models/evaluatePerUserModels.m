% #################################################################
% Experimento adicional (2026-07-24, a raíz de la pregunta del usuario:
% "¿qué pasa con un modelo específico, entrenado y testeado para la misma
% persona?"): entrena UN modelo (cabeza softmax sobre el backbone grande
% ya congelado) POR CADA usuario, usando SOLO los frames de Mes0 de ESE
% usuario (train+val combinados, para asegurar suficientes datos por
% persona -- ver nota de cobertura abajo), y lo evalúa SOLO en los frames
% de ESE MISMO usuario en Mes1..Mes6.
%
% Motivo: todos los experimentos anteriores (Tareas 1-3) usan un modelo
% POBLACIONAL (entrenado con los 19 usuarios juntos). Si el drift es un
% fenómeno de variabilidad intra-persona entre sesiones (fatiga,
% reposicionamiento del electrodo, etc.), un modelo poblacional podría
% estar promediando/enmascarando ese efecto -- este experimento aísla la
% variable "misma persona en train y test" para ver si el drift se vuelve
% más visible a ese nivel.
%
% Cobertura de usuarios en Mes0: el split train/val de
% generateSpectrogramDataset.m es por archivo/repetición (80/20
% aleatorio), NO garantiza que todos los usuarios caigan en ambos
% splits -- de hecho userID_train solo tiene 10/19 usuarios y userID_val
% solo 9/19 (se reparten distinto). Por eso aquí se combina
% train+val de Mes0 en un solo bloque por usuario (~7195 frames/usuario,
% verificado), igual que ya se hizo en evaluateMonthByMonthMatrix.m para
% Mes0.
% #################################################################

clear; clc;
rng(9);
tamano = 'grande';

%% CARGAR EL CACHE DE FEATURES
files = dir(fullfile('Models', 'FrozenFeatures_*.mat'));
esOtraVariante = contains({files.name}, {'_ann_', '_mediano_', '_pequeno_'});
files = files(~esOtraVariante);
[~, idx] = max([files.datenum]);
cacheFile = fullfile(files(idx).folder, files(idx).name);
fprintf('Cache usado: %s\n', cacheFile);
load(cacheFile, 'X_train', 'y_train', 'userID_train', 'X_val', 'y_val', 'userID_val', ...
    'X_test', 'y_test', 'userID_test', 'classNames');

numClasses = numel(classNames);
d_model = size(X_train, 1);
numMeses = numel(X_test);

%% MES0 COMBINADO (train+val) POR USUARIO
X_mes0 = [X_train, X_val];
y_mes0 = [y_train(:); y_val(:)];
userID_mes0 = [userID_train, userID_val];

allUsers = unique(userID_mes0);
numUsers = numel(allUsers);
fprintf('Usuarios con datos en Mes0: %d\n', numUsers);

%% VALIDACIÓN DE COBERTURA: cada usuario debe tener frames en Mes0 Y en todos los meses de test
minFramesMes0 = inf;
for u = 1:numUsers
    n = sum(userID_mes0 == allUsers(u));
    minFramesMes0 = min(minFramesMes0, n);
end
fprintf('Mínimo de frames en Mes0 por usuario: %d\n', minFramesMes0);
if minFramesMes0 < 100
    error('Al menos un usuario tiene muy pocos frames en Mes0 (<100) -- revisar antes de continuar.');
end

for m = 1:numMeses
    usersEsteMes = unique(userID_test{m});
    faltantes = setdiff(cellstr(allUsers), cellstr(usersEsteMes));
    if ~isempty(faltantes)
        fprintf('[AVISO] Mes%d: usuarios sin frames: %s\n', m, strjoin(faltantes, ','));
    end
end

%% ENTRENAR UN MODELO POR USUARIO
accPerUserMonth = nan(numUsers, numMeses + 1); % +1 = Mes0 (autoevaluación, referencia)
monthNames = arrayfun(@(m) sprintf('Mes%d', m), 0:numMeses, 'UniformOutput', false);

for u = 1:numUsers
    userName = allUsers(u);
    fprintf('\n========================================================================\n');
    fprintf(' Usuario %s (%d/%d)\n', userName, u, numUsers);
    fprintf('========================================================================\n');

    maskUser = (userID_mes0 == userName);
    Xu = X_mes0(:, maskUser);
    yu = y_mes0(maskUser);
    Nu = size(Xu, 2);

    % Split 80/20 dentro de los datos de ESTE usuario, solo para
    % entrenamiento + early stopping (misma lógica que
    % evaluateMonthByMonthMatrix.m).
    idxPerm = randperm(Nu);
    nTrain = floor(Nu * 0.8);
    idxTrain = idxPerm(1:nTrain);
    idxVal = idxPerm(nTrain + 1:end);

    XTrainU = Xu(:, idxTrain);
    yTrainU = yu(idxTrain);
    XValU = Xu(:, idxVal);
    yValU = yu(idxVal);

    presentClasses = unique(yTrainU);
    if numel(presentClasses) < numClasses
        fprintf('[AVISO] Usuario %s: solo %d/%d clases presentes en su split de entrenamiento -- se usa ClassWeights=1 uniforme.\n', ...
            userName, numel(presentClasses), numClasses);
        classWeightsU = ones(1, numClasses);
    else
        counts = countcats(categorical(yTrainU, classNames));
        classWeightsU = sum(counts) ./ (numClasses * max(counts, 1));
    end

    layers = [
        featureInputLayer(d_model, "Name", "context_input", "Normalization", "zscore")
        fullyConnectedLayer(numClasses, "Name", "fc_head")
        softmaxLayer("Name", "softmax_head")
        classificationLayer("Name", "classoutput_head", "Classes", classNames, "ClassWeights", classWeightsU)];

    options = trainingOptions('adam', ...
        'InitialLearnRate', 0.01, ...
        'L2Regularization', 0.001, ...
        'MaxEpochs', 200, ...
        'MiniBatchSize', 64, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', 0, ...
        'ValidationData', {XValU', yValU(:)}, ...
        'ValidationFrequency', max(1, floor(nTrain / 64)), ...
        'ValidationPatience', 10, ...
        'Plots', 'none');

    netU = trainNetwork(XTrainU', yTrainU(:), layers, options);

    % Autoevaluación en Mes0 completo de este usuario (referencia, no es
    % "válida" en el sentido de datos no vistos -- se reporta como
    % contexto, igual que la diagonal de evaluateMonthByMonthMatrix.m)
    predMes0 = classify(netU, Xu');
    accPerUserMonth(u, 1) = mean(predMes0(:) == yu(:));
    fprintf('  Mes0 (referencia, incluye datos de entrenamiento): accuracy=%.4f\n', accPerUserMonth(u, 1));

    % Evaluación en Mes1..MesN, SOLO frames de este mismo usuario
    for m = 1:numMeses
        maskUserMes = (userID_test{m} == userName);
        if ~any(maskUserMes)
            fprintf('  Mes%d: sin frames para este usuario, se omite.\n', m);
            continue;
        end
        XmU = X_test{m}(:, maskUserMes);
        ymU = y_test{m}(maskUserMes);
        predMU = classify(netU, XmU');
        accPerUserMonth(u, m + 1) = mean(predMU(:) == ymU(:));
        fprintf('  Mes%d: accuracy=%.4f (n=%d frames de este usuario)\n', m, accPerUserMonth(u, m + 1), numel(ymU));
    end
end

%% GUARDAR RESULTADOS
if ~exist('Models', 'dir')
    mkdir('Models');
end
resultsFile = sprintf('Models/PerUserModels_%s_%s.mat', tamano, datestr(now, 'dd-mm-yyyy_HH-MM-ss'));
save(resultsFile, 'accPerUserMonth', 'allUsers', 'monthNames', 'classNames', 'tamano', 'cacheFile');
fprintf('\nResultados guardados en: %s\n', resultsFile);

%% RESUMEN: media +/- SD entre usuarios, por mes (modelos PER-USUARIO)
fprintf('\n--- Resumen: accuracy media +/- SD entre usuarios (modelos per-usuario) ---\n');
mediaPorMes = mean(accPerUserMonth, 1, 'omitnan') * 100;
sdPorMes = std(accPerUserMonth, 0, 1, 'omitnan') * 100;
for m = 1:numel(monthNames)
    fprintf('%-10s media=%.2f%%  SD=%.2f\n', monthNames{m}, mediaPorMes(m), sdPorMes(m));
end

caidaAbs = mediaPorMes(2) - mediaPorMes(end); % Mes1 -> Mes6(o el último mes válido)
caidaRel = (caidaAbs / mediaPorMes(2)) * 100;
fprintf('\nCaída Mes1->%s (modelos PER-USUARIO): %.2f pp (%.2f%% relativo)\n', monthNames{end}, caidaAbs, caidaRel);

%% FIGURA: comparación media+SD, modelo per-usuario vs. modelo poblacional
% Carga la accuracy por mes del modelo POBLACIONAL (Condición A, ya
% guardada) para comparar lado a lado en la misma figura.
softmaxFiles = dir(fullfile('Models', 'SoftmaxHead_Results_*.mat'));
esOtraVariante2 = contains({softmaxFiles.name}, {'_ann_', '_mediano_', '_pequeno_'});
softmaxFiles = softmaxFiles(~esOtraVariante2);
[~, idxSoft] = max([softmaxFiles.datenum]);
poblacional = load(fullfile(softmaxFiles(idxSoft).folder, softmaxFiles(idxSoft).name), 'accByMonth', 'monthLabels');

outDir = 'FigurasReales';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

fig = figure('Visible', 'off', 'Position', [100 100 850 550]);
hold on;
xVals = 1:numel(monthNames);
errorbar(xVals, mediaPorMes, sdPorMes, '-o', 'Color', [0.85 0.33 0.10], 'LineWidth', 2, ...
    'MarkerFaceColor', [0.85 0.33 0.10], 'CapSize', 8, 'DisplayName', 'Modelo POR USUARIO (train+test misma persona)');
xValsPoblacional = 1:numel(poblacional.accByMonth);
plot(xValsPoblacional, poblacional.accByMonth * 100, '-s', 'Color', [0.20 0.35 0.60], 'LineWidth', 2, ...
    'MarkerFaceColor', [0.20 0.35 0.60], 'DisplayName', 'Modelo POBLACIONAL (19 usuarios juntos)');
hold off;
grid on;
xticks(xVals);
xticklabels(monthNames);
ylabel('Accuracy (%)');
xlabel('Sesión');
title('Modelo per-usuario vs. modelo poblacional: ¿el drift es más visible a nivel individual?');
legend('Location', 'southwest');
ylim([50 100]);
exportgraphics(fig, fullfile(outDir, 'peruser_vs_poblacional.png'), 'Resolution', 200);
close(fig);

fprintf('\nFigura guardada en: %s\n', fullfile(pwd, outDir, 'peruser_vs_poblacional.png'));
