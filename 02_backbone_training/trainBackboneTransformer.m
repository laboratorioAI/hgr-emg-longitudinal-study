% #################################################################
% PASO 1: ENTRENAMIENTO DEL BACKBONE (CNN + TRANSFORMER)
% Arquitectura basada en Barona López et al. (2024) [CNN Inception]
% y Macías et al. (2025) "HGREncoder" [Transformer: 8 heads, 128 canales]
% Ajustado para dataset reducido (19 usuarios vs 306 en la referencia)
%
% PARAMETRIZADO EN TAMAÑO (Tarea 2 / Experimento 2B, plan de respuesta a
% revisores 2026-07-22..23): 'grande' (arquitectura original, sin cambios
% de comportamiento respecto a versiones previas de este script),
% 'mediano', 'pequeno'. La reducción es proporcional en todo el backbone
% (canales por rama Inception, número de bloques Inception, d_model,
% heads, expansión FFN) para aislar el efecto de "tamaño total" sin
% cambiar el tipo de arquitectura -- ver Tarea 2 / Experimento 2B del
% plan de trabajo para la tabla completa de valores y su justificación.
%
% Reducción de bloques Inception (misma topología recortada, no una
% topología nueva):
%   grande  (6 bloques): 1a, 1b, 1c(+skip 1a), 1d, 1e(+skip 1c), 1f
%   mediano (4 bloques): 1a, 1b, 1c(+skip 1a), 1f
%   pequeno (2 bloques): 1a, 1f  (sin residual: no hay profundidad
%                                 suficiente para una skip-connection
%                                 con sentido)
% El bloque 1f (cierre con crossChannelNormalizationLayer) se conserva
% siempre como último bloque, para no alterar el punto de transición
% CNN->Transformer.
% #################################################################

function trainBackboneTransformer(tamano, soloVerificar)

if nargin < 1
    tamano = 'grande';
end
if nargin < 2
    soloVerificar = false;
end
tamano = validatestring(tamano, {'grande', 'mediano', 'pequeno', 'micro', 'nano'});

clearvars -except tamano soloVerificar; clc;

%% SEMILLA FIJA (reproducibilidad)
rng(9); % misma semilla que usa Shared.getUsers, para consistencia

%% CONFIGURACIÓN POR TAMAÑO
cfg = getSizeConfig(tamano);
fprintf('=== Backbone tamaño: %s ===\n', tamano);
fprintf('  Canales por rama Inception: %d\n', cfg.branchChannels);
fprintf('  Canales de reducción 1x1 (antes de 3x3/5x5): %d\n', cfg.reduceChannels);
fprintf('  Bloques Inception: %s\n', strjoin(cfg.blocks, ','));
fprintf('  d_model=%d  heads=%d  FFN expand=%d\n', cfg.d_model, cfg.heads, cfg.d_model * cfg.ffnMult);

if soloVerificar
    % Modo smoke-test: construye el grafo con un inputSize/numClasses
    % representativos (no requiere los datastores reales) y verifica con
    % analyzeNetwork que la topología es válida para este tamaño, sin
    % entrenar. Usado por test_smoke_backbone_sizes.m.
    inputSizeTest = [13, 24, 1]; % [frecuencias, ventana STFT, canales] tipico de Shared
    numClassesTest = 6;
    classNamesTest = categorical(1:numClassesTest);
    classWeightsTest = ones(1, numClassesTest);
    lgraph = setNeuralNetworkArchitecture(inputSizeTest, numClassesTest, cfg.heads, cfg.d_model, classNamesTest, classWeightsTest, cfg);
    analyzeNetwork(lgraph);
    % Nota: el conteo EXACTO de parámetros entrenables requiere que
    % Weights/Bias estén resueltos, lo cual solo ocurre tras entrenar
    % (trainNetwork) o ensamblar con pesos ya inicializados
    % (assembleNetwork rechaza pesos vacíos -- no sirve para esto sin
    % inicializarlos a mano). Por eso el conteo real se hace en el flujo
    % de entrenamiento (después de trainNetwork), no aquí. Este modo solo
    % valida que la topología conecta sin error.
    fprintf('[soloVerificar] %s: grafo válido (verificado con analyzeNetwork).\n', tamano);
    return;
end

%% SET DATASTORES PATHS
dataDirTraining = fullfile('DatastoresTran', 'training');
dataDirValidation = fullfile('DatastoresTran', 'validation');
if Shared.includeTesting
    dataDirTesting = fullfile('DatastoresTran', 'test');
end

%% THE DATASTORES ARE CREATED
withNoGesture = true;
classes = Shared.setNoGestureUse(withNoGesture);

trainingDatastore = SpectrogramDatastore(dataDirTraining);
validationDatastore = SpectrogramDatastore(dataDirValidation);
if Shared.includeTesting
    testingDatastore = SpectrogramDatastore(dataDirTesting);
end

clear dataDirTraining dataDirValidation withNoGesture

%% THE INPUT DIMENSIONS ARE DEFINED
inputSize = trainingDatastore.FrameDimensions;

%% DEFINE THE AMOUNT OF DATA
trainingDatastore = setDataAmount(trainingDatastore, 1);
validationDatastore = setDataAmount(validationDatastore, 1);
if Shared.includeTesting
    testingDatastore = setDataAmount(testingDatastore, 1);
end

%% DATA SUMMARY
numTrainingSamples = ['Training samples: ', num2str(trainingDatastore.NumObservations)];
numValidationSamples = ['Validation samples: ', num2str(validationDatastore.NumObservations)];
if Shared.includeTesting
    numTestingSamples = ['Testing samples: ', num2str(testingDatastore.NumObservations)];
    fprintf('\n%s\n%s\n%s\n', numTrainingSamples, numValidationSamples, numTestingSamples);
else
    fprintf('\n%s\n%s\n', numTrainingSamples, numValidationSamples);
end
clear numTrainingSamples numValidationSamples numTestingSamples

%% CLASS WEIGHTS (compensar desbalance de etiquetas por frame, no por archivo)
classNames = Shared.setNoGestureUse(true);
classWeights = computeClassWeights(trainingDatastore, classNames);
fprintf('\nPesos de clase (compensacion de desbalance por frame):\n');
for i = 1:length(classNames)
    fprintf('  %-10s peso=%.4f\n', classNames(i), classWeights(i));
end

%% NEURAL NETWORK ARCHITECTURE
numClasses = trainingDatastore.NumClasses;
d_model = cfg.d_model;
heads = cfg.heads;

lgraph = setNeuralNetworkArchitecture(inputSize, numClasses, heads, d_model, classNames, classWeights, cfg);
clear numClasses

%% TRAINING OPTIONS
miniBatchSize = 64;
maxEpochs = 10;   % consistente con convergencia reportada en Macías et al. (2025)

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.0005, ...
    'L2Regularization', 0.005, ...            % ACTIVADA: crítica dado el tamaño reducido del dataset
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor', 0.2, ...
    'LearnRateDropPeriod', 5, ...
    'ExecutionEnvironment','auto', ...
    'GradientThreshold', 1, ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Verbose', 0, ...
    'ValidationData', validationDatastore, ...
    'ValidationFrequency', floor(trainingDatastore.NumObservations / miniBatchSize), ...
    'ValidationPatience', 5, ...
    'Plots','training-progress');

clear maxEpochs miniBatchSize

%% NETWORK TRAINING
net = trainNetwork(trainingDatastore, lgraph, options);
clear options lgraph

%% CONTEO DE PARÁMETROS (post-entrenamiento, ya con tamaños resueltos)
numParams = countLearnableParameters(net);
fprintf('\nParámetros entrenables (%s): %d\n', tamano, numParams);

%% ACCURACY FOR EACH DATASET
[accTraining, flattenLabelsTraining] = calculateAccuracy(net, trainingDatastore);
[accValidation, flattenLabelsValidation] = calculateAccuracy(net, validationDatastore);
if Shared.includeTesting
    [accTesting, flattenLabelsTesting] = calculateAccuracy(net, testingDatastore);
end

strAccTraining = ['Training accuracy: ', num2str(accTraining)];
strAccValidation = ['Validation accuracy: ', num2str(accValidation)];
if Shared.includeTesting
    strAccTesting = ['Testing accuracy: ', num2str(accTesting)];
    fprintf('\n%s\n%s\n%s\n', strAccTraining, strAccValidation, strAccTesting);
else
    fprintf('\n%s\n%s\n', strAccTraining, strAccValidation);
end
clear accTraining accValidation accTesting strAccTraining strAccValidation strAccTesting

%% CONFUSION MATRIX FOR EACH DATASET
calculateConfusionMatrix(flattenLabelsTraining, ['training-' tamano]);
calculateConfusionMatrix(flattenLabelsValidation, ['validation-' tamano]);
if Shared.includeTesting
    calculateConfusionMatrix(flattenLabelsTesting, ['testing-' tamano]);
end

%% SAVE MODEL
if ~exist('Models', 'dir')
    mkdir('Models');
end
nombreModelo = sprintf('Models/Backbone_CNNTransformer_%s_dmodel%d_heads%d_%s.mat', ...
    tamano, d_model, heads, datestr(now,'dd-mm-yyyy_HH-MM-ss'));
save(nombreModelo, 'net', 'd_model', 'heads', 'tamano', 'numParams', 'cfg');
fprintf('\nModelo guardado en: %s\n', nombreModelo);

%% CARACTERIZACIÓN TEMPORAL (entrenado en Mes0, evaluado en Mes1-Mes6)
if Shared.includeTesting
    numMeses = 6;
    accPerMonth = nan(numMeses, 1);
    fprintf('\n--- Desempeño a lo largo del tiempo (train=Mes0, tamano=%s) ---\n', tamano);
    for mes = 1:numMeses
        dsMes = filterByMonth(testingDatastore, mes);
        if dsMes.NumObservations == 0
            fprintf('Mes%d: sin datos, se omite.\n', mes);
            continue;
        end
        [accMes, flattenLabelsMes] = calculateAccuracy(net, dsMes);
        accPerMonth(mes) = accMes;
        fprintf('Mes%d accuracy: %.4f (n=%d secuencias)\n', mes, accMes, dsMes.NumObservations);
        calculateConfusionMatrix(flattenLabelsMes, sprintf('test-Mes%d-%s', mes, tamano));
    end

    figure('Name', ['Desempeño a lo largo del tiempo - ' tamano]);
    plot(1:numMeses, accPerMonth, '-o', 'LineWidth', 1.5);
    xlabel('Mes'); ylabel('Accuracy');
    title(sprintf('Desempeño del modelo por mes (train=Mes0, test=Mes1..Mes6) — %s', tamano));
    xticks(1:numMeses);
    grid on;

    clear numMeses mes dsMes accMes flattenLabelsMes accPerMonth
end

end


%% ======================================================
%% CONFIGURACIÓN DE TAMAÑO (grande / mediano / pequeño)
%% ======================================================
function cfg = getSizeConfig(tamano)
    switch tamano
        case 'grande'
            cfg.branchChannels = 18;
            cfg.reduceChannels = 16;
            cfg.blocks = {'1a', '1b', '1c', '1d', '1e', '1f'};
            cfg.d_model = 128;
            cfg.heads = 8;
            cfg.ffnMult = 2;
        case 'mediano'
            cfg.branchChannels = 12;
            cfg.reduceChannels = 10;
            cfg.blocks = {'1a', '1b', '1c', '1f'};
            cfg.d_model = 64;
            cfg.heads = 4;
            cfg.ffnMult = 2;
        case 'pequeno'
            cfg.branchChannels = 6;
            cfg.reduceChannels = 5;
            cfg.blocks = {'1a', '1f'};
            cfg.d_model = 32;
            cfg.heads = 2;
            cfg.ffnMult = 2;
        case 'micro'
            % Ultra-reducido (Tarea de verificacion 2026-07-27): busca un
            % umbral de capacidad donde el patron "Mes0 como valle, no pico"
            % (visto en mediano/pequeno/ann/tcn) se revierta. Misma topologia
            % minima de 2 bloques Inception que 'pequeno' (1a, 1f), reducida
            % aun mas en canales y d_model, mismo protocolo de entrenamiento
            % (10 epocas fijas) para que sea comparable con los tamanos ya
            % existentes.
            cfg.branchChannels = 3;
            cfg.reduceChannels = 3;
            cfg.blocks = {'1a', '1f'};
            cfg.d_model = 16;
            cfg.heads = 2;
            cfg.ffnMult = 2;
        case 'nano'
            % Minimo funcional: 1 canal por rama Inception (4 canales de
            % salida por bloque tras concatenar), d_model=8 con 1 sola
            % cabeza de atencion (no se puede subdividir mas d_model=8 en
            % >1 cabeza de forma limpia sin caer en d_model/heads no entero
            % por debajo de 8).
            cfg.branchChannels = 1;
            cfg.reduceChannels = 1;
            cfg.blocks = {'1a', '1f'};
            cfg.d_model = 8;
            cfg.heads = 1;
            cfg.ffnMult = 2;
    end
end


%% ======================================================
%% FUNCTION TO ESTABLISH THE NEURAL NETWORK ARCHITECTURE
%% ======================================================
function lgraph = setNeuralNetworkArchitecture(inputSize, numClasses, heads, d_model, classNames, classWeights, cfg)

    lgraph = layerGraph();
    C = cfg.branchChannels;   % canales de salida por rama Inception
    R = cfg.reduceChannels;   % canales de la reducción 1x1 previa a 3x3/5x5
    blocks = cfg.blocks;

    %% INPUT
    tempLayers = [
        sequenceInputLayer(inputSize,"Name","sequence")
        sequenceFoldingLayer("Name","seqfold")];
    lgraph = addLayers(lgraph,tempLayers);

    prevLayer = 'seqfold/out'; % salida del bloque anterior, alimenta al bloque Inception actual
    skipSource = '';           % nombre de la capa que se usará como skip-connection (si aplica)

    %% INCEPTION 1A (siempre presente, primer bloque)
    tempLayers = [
        convolution2dLayer([1 1],R,"Name","Inception_1a-3x3_reduce")
        reluLayer("Name","Inception_1a-3x3_relu_reduce")
        convolution2dLayer([3 3],C,"Name","Inception_1a-3x3","Padding",[1 1 1 1])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = convolution2dLayer([1 1],C,"Name","Inception_1a-1x1");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        maxPooling2dLayer([3 3],"Name","Inception_1a-pool","Padding",[1 1 1 1])
        convolution2dLayer([1 1],C,"Name","Inception_1a-pool_proj")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],R,"Name","Inception_1a-5x5_reduce")
        reluLayer("Name","Inception_1a-5x5_relu_reduce_2")
        convolution2dLayer([5 5],C,"Name","Inception_1a-5x5","Padding",[2 2 2 2])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        depthConcatenationLayer(4,"Name","depthcat_1a")
        reluLayer("Name","Inception_1a_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    lgraph = connectLayers(lgraph,prevLayer,"Inception_1a-3x3_reduce");
    lgraph = connectLayers(lgraph,prevLayer,"Inception_1a-1x1");
    lgraph = connectLayers(lgraph,prevLayer,"Inception_1a-pool");
    lgraph = connectLayers(lgraph,prevLayer,"Inception_1a-5x5_reduce");
    lgraph = connectLayers(lgraph,"Inception_1a-3x3","depthcat_1a/in1");
    lgraph = connectLayers(lgraph,"Inception_1a-5x5","depthcat_1a/in2");
    lgraph = connectLayers(lgraph,"Inception_1a-pool_proj","depthcat_1a/in3");
    lgraph = connectLayers(lgraph,"Inception_1a-1x1","depthcat_1a/in4");

    prevLayer = 'Inception_1a_relu';
    lastBlockOut = 'Inception_1a_relu'; % salida del último bloque Inception construido, para conectar el bloque 1f al final

    %% INCEPTION 1B (opcional)
    if ismember('1b', blocks)
        tempLayers = [
            maxPooling2dLayer([3 3],"Name","Inception_1b-pool","Padding",[1 1 1 1])
            convolution2dLayer([1 1],C,"Name","Inception_1b-pool_proj")];
        lgraph = addLayers(lgraph,tempLayers);

        tempLayers = [
            convolution2dLayer([1 1],R,"Name","Inception_1b-3x3_reduce")
            reluLayer("Name","Inception_1b-3x3_relu_reduce")
            convolution2dLayer([3 3],C,"Name","Inception_1b-3x3","Padding",[1 1 1 1])];
        lgraph = addLayers(lgraph,tempLayers);

        tempLayers = convolution2dLayer([1 1],C,"Name","Inception_1b-1x1");
        lgraph = addLayers(lgraph,tempLayers);

        tempLayers = [
            convolution2dLayer([1 1],R,"Name","Inception_1b-5x5_reduce")
            reluLayer("Name","Inception_1b-5x5_relu_reduce_2")
            convolution2dLayer([5 5],C,"Name","Inception_1b-5x5","Padding",[2 2 2 2])];
        lgraph = addLayers(lgraph,tempLayers);

        tempLayers = [
            depthConcatenationLayer(4,"Name","depthcat_1b")
            reluLayer("Name","Inception_1b")];
        lgraph = addLayers(lgraph,tempLayers);

        lgraph = connectLayers(lgraph,prevLayer,"Inception_1b-pool");
        lgraph = connectLayers(lgraph,prevLayer,"Inception_1b-3x3_reduce");
        lgraph = connectLayers(lgraph,prevLayer,"Inception_1b-1x1");
        lgraph = connectLayers(lgraph,prevLayer,"Inception_1b-5x5_reduce");
        lgraph = connectLayers(lgraph,"Inception_1b-5x5","depthcat_1b/in1");
        lgraph = connectLayers(lgraph,"Inception_1b-1x1","depthcat_1b/in2");
        lgraph = connectLayers(lgraph,"Inception_1b-3x3","depthcat_1b/in3");
        lgraph = connectLayers(lgraph,"Inception_1b-pool_proj","depthcat_1b/in4");

        prevLayer = 'Inception_1b';
        lastBlockOut = 'Inception_1b';
    end

    %% INCEPTION 1C (opcional, con skip-connection desde 1a si 1a existe y 1c está presente)
    if ismember('1c', blocks)
        tempLayers = [
            maxPooling2dLayer([3 3],"Name","Inception_1c-pool","Padding",[1 1 1 1])
            convolution2dLayer([1 1],C,"Name","Inception_1c-pool_proj")];
        lgraph = addLayers(lgraph,tempLayers);

        tempLayers = [
            convolution2dLayer([1 1],R,"Name","Inception_1c-3x3_reduce")
            reluLayer("Name","Inception_1c-3x3_relu_reduce")
            convolution2dLayer([3 3],C,"Name","Inception_1c-3x3","Padding",[1 1 1 1])];
        lgraph = addLayers(lgraph,tempLayers);

        tempLayers = [
            convolution2dLayer([1 1],R,"Name","Inception_1c-5x5_reduce")
            reluLayer("Name","Inception_1c-5x5_relu_reduce_2")
            convolution2dLayer([5 5],C,"Name","Inception_1c-5x5","Padding",[2 2 2 2])];
        lgraph = addLayers(lgraph,tempLayers);

        tempLayers = convolution2dLayer([1 1],C,"Name","Inception_1c-1x1");
        lgraph = addLayers(lgraph,tempLayers);

        tempLayers = depthConcatenationLayer(4,"Name","depthcat_1c");
        lgraph = addLayers(lgraph,tempLayers);

        tempLayers = [
            additionLayer(2,"Name","addition_1ac")
            reluLayer("Name","Inception_1c")];
        lgraph = addLayers(lgraph,tempLayers);

        lgraph = connectLayers(lgraph,prevLayer,"Inception_1c-pool");
        lgraph = connectLayers(lgraph,prevLayer,"Inception_1c-3x3_reduce");
        lgraph = connectLayers(lgraph,prevLayer,"Inception_1c-5x5_reduce");
        lgraph = connectLayers(lgraph,prevLayer,"Inception_1c-1x1");
        lgraph = connectLayers(lgraph,"Inception_1c-1x1","depthcat_1c/in1");
        lgraph = connectLayers(lgraph,"Inception_1c-3x3","depthcat_1c/in2");
        lgraph = connectLayers(lgraph,"Inception_1c-5x5","depthcat_1c/in3");
        lgraph = connectLayers(lgraph,"Inception_1c-pool_proj","depthcat_1c/in4");

        % Skip-connection: usa la salida de Inception_1a_relu (siempre existe)
        lgraph = connectLayers(lgraph,"Inception_1a_relu","addition_1ac/in1");
        lgraph = connectLayers(lgraph,"depthcat_1c","addition_1ac/in2");

        prevLayer = 'Inception_1c';
        lastBlockOut = 'Inception_1c';
    end

    %% INCEPTION 1D (opcional, solo variante grande)
    if ismember('1d', blocks)
        tempLayers = [
            convolution2dLayer([1 1],R,"Name","Inception_1d-5x5_reduce")
            reluLayer("Name","Inception_1d-5x5_relu_reduce_2")
            convolution2dLayer([5 5],C,"Name","Inception_1d-5x5","Padding",[2 2 2 2])];
        lgraph = addLayers(lgraph,tempLayers);

        tempLayers = [
            maxPooling2dLayer([3 3],"Name","Inception_1d-pool","Padding",[1 1 1 1])
            convolution2dLayer([1 1],C,"Name","Inception_1d-pool_proj")];
        lgraph = addLayers(lgraph,tempLayers);

        tempLayers = convolution2dLayer([1 1],C,"Name","Inception_1d-1x1");
        lgraph = addLayers(lgraph,tempLayers);

        tempLayers = [
            convolution2dLayer([1 1],R,"Name","Inception_1d-3x3_reduce")
            reluLayer("Name","Inception_1d-3x3_relu_reduce")
            convolution2dLayer([3 3],C,"Name","Inception_1d-3x3","Padding",[1 1 1 1])];
        lgraph = addLayers(lgraph,tempLayers);

        tempLayers = [
            depthConcatenationLayer(4,"Name","depthcat_1d")
            reluLayer("Name","Inception_1d")];
        lgraph = addLayers(lgraph,tempLayers);

        lgraph = connectLayers(lgraph,prevLayer,"Inception_1d-5x5_reduce");
        lgraph = connectLayers(lgraph,prevLayer,"Inception_1d-pool");
        lgraph = connectLayers(lgraph,prevLayer,"Inception_1d-1x1");
        lgraph = connectLayers(lgraph,prevLayer,"Inception_1d-3x3_reduce");
        lgraph = connectLayers(lgraph,"Inception_1d-1x1","depthcat_1d/in1");
        lgraph = connectLayers(lgraph,"Inception_1d-3x3","depthcat_1d/in2");
        lgraph = connectLayers(lgraph,"Inception_1d-5x5","depthcat_1d/in3");
        lgraph = connectLayers(lgraph,"Inception_1d-pool_proj","depthcat_1d/in4");

        prevLayer = 'Inception_1d';
        lastBlockOut = 'Inception_1d';
    end

    %% INCEPTION 1E (opcional, solo variante grande, con skip desde 1c)
    if ismember('1e', blocks)
        tempLayers = [
            maxPooling2dLayer([3 3],"Name","Inception_1e-pool","Padding",[1 1 1 1])
            convolution2dLayer([1 1],C,"Name","Inception_1e-pool_proj")];
        lgraph = addLayers(lgraph,tempLayers);

        tempLayers = convolution2dLayer([1 1],C,"Name","Inception_1e-1x1");
        lgraph = addLayers(lgraph,tempLayers);

        tempLayers = [
            convolution2dLayer([1 1],R,"Name","Inception_1e-5x5_reduce")
            reluLayer("Name","Inception_1e-5x5_relu_reduce_2")
            convolution2dLayer([5 5],C,"Name","Inception_1e-5x5","Padding",[2 2 2 2])];
        lgraph = addLayers(lgraph,tempLayers);

        tempLayers = [
            convolution2dLayer([1 1],R,"Name","Inception_1e-3x3_reduce")
            reluLayer("Name","Inception_1e-3x3_relu_reduce")
            convolution2dLayer([3 3],C,"Name","Inception_1e-3x3","Padding",[1 1 1 1])];
        lgraph = addLayers(lgraph,tempLayers);

        tempLayers = depthConcatenationLayer(4,"Name","depthcat_1e");
        lgraph = addLayers(lgraph,tempLayers);

        tempLayers = [
            additionLayer(2,"Name","addition_1ce")
            reluLayer("Name","Inception_1e")];
        lgraph = addLayers(lgraph,tempLayers);

        lgraph = connectLayers(lgraph,prevLayer,"Inception_1e-pool");
        lgraph = connectLayers(lgraph,prevLayer,"Inception_1e-1x1");
        lgraph = connectLayers(lgraph,prevLayer,"Inception_1e-5x5_reduce");
        lgraph = connectLayers(lgraph,prevLayer,"Inception_1e-3x3_reduce");
        lgraph = connectLayers(lgraph,"Inception_1e-1x1","depthcat_1e/in1");
        lgraph = connectLayers(lgraph,"Inception_1e-3x3","depthcat_1e/in2");
        lgraph = connectLayers(lgraph,"Inception_1e-5x5","depthcat_1e/in3");
        lgraph = connectLayers(lgraph,"Inception_1e-pool_proj","depthcat_1e/in4");

        % Skip-connection: usa la salida de Inception_1c (requiere 1c presente)
        lgraph = connectLayers(lgraph,"Inception_1c","addition_1ce/in2");
        lgraph = connectLayers(lgraph,"depthcat_1e","addition_1ce/in1");

        prevLayer = 'Inception_1e';
        lastBlockOut = 'Inception_1e';
    end

    %% INCEPTION 1F (siempre presente, bloque de cierre -> crossnorm)
    tempLayers = [
        convolution2dLayer([1 1],R,"Name","Inception_1f-3x3_reduce")
        reluLayer("Name","Inception_1f-3x3_relu_reduce")
        convolution2dLayer([3 3],C,"Name","Inception_1f-3x3","Padding",[1 1 1 1])
        reluLayer("Name","Inception_1f-3x3_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        maxPooling2dLayer([3 3],"Name","Inception_1f-pool","Padding",[1 1 1 1])
        convolution2dLayer([1 1],C,"Name","Inception_1f-pool_proj")
        reluLayer("Name","Inception_1f-relu-pool_proj")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],C,"Name","Inception_1f-1x1")
        reluLayer("Name","Inception_1f-1x1_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],R,"Name","Inception_1f-5x5_reduce")
        reluLayer("Name","Inception_1f-5x5_relu_reduce_2")
        convolution2dLayer([5 5],C,"Name","Inception_1f-5x5","Padding",[2 2 2 2])
        reluLayer("Name","Inception_1f-5x5_relu")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        depthConcatenationLayer(4,"Name","depthcat_1f")
        crossChannelNormalizationLayer(5,"Name","crossnorm_1")];
    lgraph = addLayers(lgraph,tempLayers);

    lgraph = connectLayers(lgraph,lastBlockOut,"Inception_1f-3x3_reduce");
    lgraph = connectLayers(lgraph,lastBlockOut,"Inception_1f-pool");
    lgraph = connectLayers(lgraph,lastBlockOut,"Inception_1f-1x1");
    lgraph = connectLayers(lgraph,lastBlockOut,"Inception_1f-5x5_reduce");
    lgraph = connectLayers(lgraph,"Inception_1f-1x1_relu","depthcat_1f/in1");
    lgraph = connectLayers(lgraph,"Inception_1f-3x3_relu","depthcat_1f/in2");
    lgraph = connectLayers(lgraph,"Inception_1f-5x5_relu","depthcat_1f/in3");
    lgraph = connectLayers(lgraph,"Inception_1f-relu-pool_proj","depthcat_1f/in4");

    %% TRANSFORMER BLOCK
    tempLayers = [
        sequenceUnfoldingLayer("Name","sequnfold")
        flattenLayer("Name","flatten")
        fullyConnectedLayer(d_model,"Name","fc")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = sinusoidalPositionEncodingLayer(d_model,"Name","positionencode");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        additionLayer(2,"Name","addition_attention")
        selfAttentionLayer(heads, d_model,"Name","selfattention")
        dropoutLayer(0.4,"Name","dropout_1")

        additionLayer(2, "Name", "addition_ffn")
        fullyConnectedLayer(d_model * cfg.ffnMult, "Name", "ffn_expand")
        reluLayer("Name", "ffn_relu")
        fullyConnectedLayer(d_model, "Name", "ffn_compress")
        dropoutLayer(0.4,"Name","dropout_2")

        fullyConnectedLayer(numClasses,"Name","fc_final")
        softmaxLayer("Name","softmax")
        classificationLayer("Name","classoutput","Classes",classNames,"ClassWeights",classWeights)];
    lgraph = addLayers(lgraph,tempLayers);

    %% CNN -> TRANSFORMER
    lgraph = connectLayers(lgraph,"seqfold/miniBatchSize","sequnfold/miniBatchSize");
    lgraph = connectLayers(lgraph,"crossnorm_1","sequnfold/in");

    %% POSITIONAL ENCODING
    lgraph = connectLayers(lgraph,"fc","positionencode");

    %% RESIDUAL CONNECTIONS (TRANSFORMER)
    lgraph = connectLayers(lgraph,"fc","addition_attention/in1");
    lgraph = connectLayers(lgraph,"positionencode","addition_attention/in2");
    lgraph = connectLayers(lgraph,"dropout_1","addition_ffn/in2");

end


%% ======================================================
%% FUNCTION TO COUNT LEARNABLE PARAMETERS (post-entrenamiento, exacto)
%% ======================================================
function total = countLearnableParameters(net)
    % Se llama DESPUÉS de trainNetwork (o tras construir la red con un
    % forward-pass real), momento en el cual net.Layers(i).Weights/Bias
    % ya están resueltos con sus tamaños reales (antes de entrenar,
    % dependen del tamaño de entrada y aparecen como 0x0). Recorre cada
    % capa y suma numel(...) de cualquier propiedad de pesos/sesgo no
    % vacía -- exacto, no una aproximación analítica.
    total = 0;
    for i = 1:numel(net.Layers)
        layer = net.Layers(i);
        props = properties(layer);
        for p = 1:numel(props)
            name = props{p};
            if any(strcmp(name, {'Weights', 'Bias', 'InputWeights', 'RecurrentWeights'}))
                val = layer.(name);
                if isnumeric(val)
                    total = total + numel(val);
                end
            end
        end
    end
end


%% ======================================================
%% FUNCTION TO CALCULATE ACCURACY OF A DATASTORE
%% ======================================================
function [accuracy, flattenLabels] = calculateAccuracy(net, datastore)
    realVsPredData = cell(datastore.NumObservations, 2);
    datastore.MiniBatchSize = 1;

    totalLabels = 0;
    while hasdata(datastore)
        position = datastore.CurrentFileIndex;
        data = read(datastore);
        labels = data.labelsSequences;
        sequence = data.sequences;
        labelsPred = classify(net,sequence);
        realVsPredData(position, :) = [labels, labelsPred];
        totalLabels = totalLabels + length(labels{1,1});
    end

    flattenLabels = cell(totalLabels,2);
    idx = 0;
    for i = 1:length(realVsPredData)
        labels = realVsPredData{i, 1};
        labelsPred = realVsPredData{i, 2};
        for j = 1:length(labels)
            flattenLabels{idx+j, 1} = char(labels(1,j));
            flattenLabels{idx+j, 2} = char(labelsPred(1, j));
        end
        idx = idx + length(labels);
    end

    matches = 0;
    for i = 1:length(flattenLabels)
        if isequal(flattenLabels{i, 1}, flattenLabels{i, 2})
            matches = matches + 1;
        end
    end

    accuracy = matches / length(flattenLabels);
    reset(datastore);
end


%% ======================================================
%% FUNCTION TO CALCULATE AND PLOT A CONFUSION MATRIX
%% ======================================================
function calculateConfusionMatrix(flattenLabels, datasetName)
    classes = categorical(Shared.setNoGestureUse(true));

    realLabels = categorical(flattenLabels(:,1), Shared.setNoGestureUse(true));
    predLabels = categorical(flattenLabels(:,2), Shared.setNoGestureUse(true));

    confusionMatrix = confusionmat(realLabels, predLabels, 'Order', classes);
    figure('Name', ['Confusion Matrix - ' datasetName])
        matrixChart = confusionchart(confusionMatrix, classes);
        matrixChart.ColumnSummary = 'column-normalized';
        matrixChart.RowSummary = 'row-normalized';
        matrixChart.Title = ['Hand gestures - ' datasetName];
        sortClasses(matrixChart,classes);
end


%% ======================================================
%% FUNCTION TO COMPUTE PER-FRAME CLASS WEIGHTS
%% ======================================================
function weights = computeClassWeights(datastore, classNames)
    numClasses = length(classNames);
    counts = zeros(1, numClasses);

    reset(datastore);
    while hasdata(datastore)
        data = read(datastore);
        for i = 1:height(data)
            counts = counts + countcats(data.labelsSequences{i});
        end
    end
    reset(datastore);

    totalCount = sum(counts);
    weights = totalCount ./ (numClasses * counts);
end
