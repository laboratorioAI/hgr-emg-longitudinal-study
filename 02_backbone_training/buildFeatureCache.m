% #################################################################
% PASO 2a: EXTRACCIÓN DE FEATURES CONGELADAS (CNN + Transformer)
% Congela el backbone hasta 'dropout_2' (128-d, salida de atención+FFN,
% justo antes de fc_final) y cachea el vector de contexto por frame para
% Mes0 (training/validation) y Mes1..Mes6 (test, separado por mes con
% filterByMonth). Este cache se reutiliza tanto por trainSoftmaxHead.m
% como por context_bandit.m, para que ambos paradigmas partan exactamente
% de los mismos features y no se recompute el forward pass del backbone
% dos veces.
%
% PARAMETRIZADO POR TAMAÑO (Tarea 2 / Experimento 2B): recibe 'grande'
% (default, comportamiento sin cambios), 'mediano' o 'pequeno', y busca
% específicamente el backbone de ESE tamaño (por sufijo en el nombre de
% archivo, no simplemente "el más reciente de cualquier tamaño") para que
% correr esto para la variante mediana/pequeña nunca recoja por accidente
% el backbone grande ya entrenado. El cache resultante también lleva el
% sufijo de tamaño, para que trainSoftmaxHead.m / context_bandit.m /
% context_bandit_adaptive.m puedan apuntar exactamente a la variante que
% corresponde.
%
% 'tcn' (2026-07-24, experimento de arquitectura alternativa): el backbone
% TCN (trainBackboneTCN.m) guarda su modelo con un patrón de nombre
% distinto (Models/Backbone_TCN_*.mat, sin la variable 'd_model' -- usa
% 'numTcnChannels' en su lugar) pero también tiene una capa 'dropout_2'
% (128-d, ver setTCNArchitecture en trainBackboneTCN.m), así que el resto
% de este script funciona sin cambios.
% #################################################################

function buildFeatureCache(tamano)

if nargin < 1
    tamano = 'grande';
end
tamano = validatestring(tamano, {'grande', 'mediano', 'pequeno', 'tcn', 'micro', 'nano'});

clearvars -except tamano; clc;
rng(9);

FEATURE_LAYER = 'dropout_2';

%% CARGAR EL BACKBONE MÁS RECIENTE DE ESTE TAMAÑO
if strcmp(tamano, 'tcn')
    backboneFiles = dir(fullfile('Models', 'Backbone_TCN_*.mat'));
    if isempty(backboneFiles)
        error('No se encontró ningún Models/Backbone_TCN_*.mat. Corre trainBackboneTCN() primero.');
    end
    [~, newestIdx] = max([backboneFiles.datenum]);
    backboneFile = fullfile(backboneFiles(newestIdx).folder, backboneFiles(newestIdx).name);
    fprintf('Backbone usado (%s): %s\n', tamano, backboneFile);
    loaded = load(backboneFile, 'net');
    net = loaded.net;
    d_model = 128; % dimensión fija de la capa dropout_2 en setTCNArchitecture (fc_projection)
else
    backboneFiles = dir(fullfile('Models', sprintf('Backbone_CNNTransformer_%s_*.mat', tamano)));
    if isempty(backboneFiles)
        error('No se encontró ningún Models/Backbone_CNNTransformer_%s_*.mat. Corre trainBackboneTransformer(''%s'') primero.', tamano, tamano);
    end
    [~, newestIdx] = max([backboneFiles.datenum]);
    backboneFile = fullfile(backboneFiles(newestIdx).folder, backboneFiles(newestIdx).name);
    fprintf('Backbone usado (%s): %s\n', tamano, backboneFile);
    loaded = load(backboneFile, 'net', 'd_model');
    net = loaded.net;
    d_model = loaded.d_model;
end

%% DATASTORES
classNames = Shared.setNoGestureUse(true);
trainingDatastore = SpectrogramDatastore(fullfile('DatastoresTran', 'training'));
validationDatastore = SpectrogramDatastore(fullfile('DatastoresTran', 'validation'));
testingDatastore = SpectrogramDatastore(fullfile('DatastoresTran', 'test'));

%% EXTRACCIÓN MES0 (TRAIN / VALIDATION)
fprintf('\nExtrayendo features de Mes0 (training)...\n');
[X_train, y_train, userID_train] = extractFeatures(net, trainingDatastore, FEATURE_LAYER, d_model);
fprintf('Extrayendo features de Mes0 (validation)...\n');
[X_val, y_val, userID_val] = extractFeatures(net, validationDatastore, FEATURE_LAYER, d_model);

%% EXTRACCIÓN MES1..MES6 (TEST, POR MES)
numMeses = 6;
X_test = cell(1, numMeses);
y_test = cell(1, numMeses);
userID_test = cell(1, numMeses);
for m = 1:numMeses
    dsMes = filterByMonth(testingDatastore, m);
    fprintf('Extrayendo features de Mes%d (%d secuencias)...\n', m, dsMes.NumObservations);
    [X_test{m}, y_test{m}, userID_test{m}] = extractFeatures(net, dsMes, FEATURE_LAYER, d_model);
end

%% GUARDAR CACHE
if ~exist('Models', 'dir')
    mkdir('Models');
end
cacheFile = sprintf('Models/FrozenFeatures_%s_%s.mat', tamano, datestr(now, 'dd-mm-yyyy_HH-MM-ss'));
save(cacheFile, 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', ...
    'userID_train', 'userID_val', 'userID_test', ...
    'classNames', 'FEATURE_LAYER', 'backboneFile', 'tamano', '-v7.3');
fprintf('\nCache de features guardado en: %s\n', cacheFile);

end


%% ======================================================
%% FUNCTION TO EXTRACT FROZEN FEATURES FROM A DATASTORE
%% ======================================================
function [X, y, userID] = extractFeatures(net, datastore, featureLayer, d_model)
    reset(datastore);
    datastore.MiniBatchSize = 1;
    numObs = datastore.NumObservations;

    Xcell = cell(numObs, 1);
    ycell = cell(numObs, 1);
    userIDcell = cell(numObs, 1);
    idx = 0;

    while hasdata(datastore)
        idx = idx + 1;

        % El nombre de archivo de la secuencia que se va a leer a
        % continuación (ej. "Mes3_user12_..._rep2_seq.mat") codifica el
        % usuario de origen. Se captura ANTES de leer porque read()
        % incrementa CurrentFileIndex al terminar.
        currentFile = datastore.Datastore.Files{datastore.CurrentFileIndex};
        [~, currentName, ~] = fileparts(currentFile);
        userTok = regexp(currentName, '_(user\d+)_', 'tokens', 'once');
        if isempty(userTok)
            error('No se pudo extraer el usuario del nombre de archivo: %s', currentName);
        end
        userName = userTok{1};

        data = read(datastore);
        sequence = data.sequences{1};
        labelsSeq = data.labelsSequences{1};

        feat = activations(net, sequence, featureLayer, 'OutputAs', 'channels');
        if iscell(feat), feat = feat{1}; end % la red con folding/unfolding devuelve activations envueltas en celda
        feat = squeeze(feat); % esperado: d_model x numFrames
        if size(feat, 1) ~= d_model && size(feat, 2) == d_model
            feat = feat'; % venía transpuesta
        end

        Xcell{idx} = feat;
        ycell{idx} = labelsSeq;
        userIDcell{idx} = repmat(string(userName), 1, numel(labelsSeq));

        if mod(idx, 500) == 0
            fprintf('  ... %d/%d secuencias procesadas\n', idx, numObs);
        end
    end

    X = [Xcell{:}];
    y = [ycell{:}];
    userID = [userIDcell{:}];
    reset(datastore);
end
