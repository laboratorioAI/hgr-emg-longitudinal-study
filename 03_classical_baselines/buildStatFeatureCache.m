% #################################################################
% Experimento 2A: construye el "cache" de features clásicos directamente
% desde DatastoresStatFeatures/<split>/<gesto>/*.mat (generado por
% generateStatFeatureDataset.m), en el MISMO formato (X_train, y_train,
% X_val, y_val, X_test{1..6}, y_test{1..6}, userID_*) que usa
% buildFeatureCache.m para el backbone CNN+Transformer -- así
% trainSoftmaxHead.m, context_bandit.m y context_bandit_adaptive.m pueden
% consumir este cache sin ningún cambio, simplemente apuntando al tamaño
% 'ann' en vez de 'grande'/'mediano'/'pequeno'.
%
% A diferencia de buildFeatureCache.m, aquí NO hay backbone que congelar:
% los features estadísticos (MAV/RMS/SD/energía) ya son "la
% representación congelada" en sí -- no se aprende ninguna representación,
% por diseño (esa es la esencia del Experimento 2A: features hechos a
% mano, no aprendidos). Por eso se lee directo de disco, sin red neuronal
% de por medio, en vez de reusar SpectrogramDatastore (que asume
% CNN+secuencias, innecesario para vectores tabulares).
% #################################################################

clear; clc;
rng(9);

FEATURE_LAYER = 'features_clasicos_MAV_RMS_SD_energia';
tamano = 'ann';

dataDir = fullfile('DatastoresStatFeatures');
if ~exist(dataDir, 'dir')
    error('No se encontró %s. Corre generateStatFeatureDataset.m primero.', dataDir);
end

classNames = Shared.setNoGestureUse(true);

fprintf('Cargando split training...\n');
[X_train, y_train, userID_train] = loadSplit(fullfile(dataDir, 'training'), classNames);
fprintf('Cargando split validation...\n');
[X_val, y_val, userID_val] = loadSplit(fullfile(dataDir, 'validation'), classNames);

fprintf('Cargando split test (por mes)...\n');
numMeses = 6;
X_test = cell(1, numMeses);
y_test = cell(1, numMeses);
userID_test = cell(1, numMeses);
for m = 1:numMeses
    fprintf('  Mes%d...\n', m);
    [X_test{m}, y_test{m}, userID_test{m}] = loadSplit(fullfile(dataDir, 'test'), classNames, sprintf('Mes%d_', m));
end

if ~exist('Models', 'dir')
    mkdir('Models');
end
cacheFile = sprintf('Models/FrozenFeatures_%s_%s.mat', tamano, datestr(now, 'dd-mm-yyyy_HH-MM-ss'));
save(cacheFile, 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', ...
    'userID_train', 'userID_val', 'userID_test', ...
    'classNames', 'FEATURE_LAYER', 'tamano', '-v7.3');
fprintf('\nCache de features clásicos (Experimento 2A) guardado en: %s\n', cacheFile);


%% ======================================================
%% FUNCTION: cargar todos los .mat de un split (opcionalmente filtrado por prefijo de mes)
%% ======================================================
function [X, y, userID] = loadSplit(splitDir, classNames, monthPrefix)
    if nargin < 3
        monthPrefix = '';
    end

    gestures = cellstr(classNames);
    Xcell = {};
    ycell = {};
    userIDcell = {};
    totalFiles = 0;

    for g = 1:numel(gestures)
        gestureDir = fullfile(splitDir, gestures{g});
        if ~exist(gestureDir, 'dir')
            continue;
        end
        files = dir(fullfile(gestureDir, '*.mat'));
        for f = 1:numel(files)
            fileName = files(f).name;
            if ~isempty(monthPrefix) && ~startsWith(fileName, monthPrefix)
                continue;
            end

            userTok = regexp(fileName, '_(user\d+)_', 'tokens', 'once');
            if isempty(userTok)
                error('No se pudo extraer el usuario del nombre de archivo: %s', fileName);
            end
            userName = userTok{1};

            loaded = load(fullfile(gestureDir, fileName), 'data');
            seqData = loaded.data.sequenceData; % Nx3: {featVec, label, timestamp}
            numFrames = size(seqData, 1);

            featMat = [seqData{:, 1}]; % 32 x numFrames (cada featVec es 32x1)
            labelsSeq = categorical(seqData(:, 2), gestures)';

            Xcell{end+1} = featMat; %#ok<AGROW>
            ycell{end+1} = labelsSeq; %#ok<AGROW>
            userIDcell{end+1} = repmat(string(userName), 1, numFrames); %#ok<AGROW>

            totalFiles = totalFiles + 1;
            if mod(totalFiles, 500) == 0
                fprintf('    ... %d archivos procesados\n', totalFiles);
            end
        end
    end

    X = [Xcell{:}];
    y = [ycell{:}];
    userID = [userIDcell{:}];
end
