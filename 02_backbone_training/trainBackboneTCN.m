% #################################################################
% Experimento adicional (2026-07-24, a raíz de la pregunta del usuario:
% "¿existe alguna arquitectura que no use CNN+Transformer, para probar si
% la combinación actual es la que mitiga el drift?"): backbone TCN
% (Temporal Convolutional Network, Bai, Kolter & Koutník 2018), una
% arquitectura GENUINAMENTE distinta -- sin bloques Inception, sin
% self-attention -- que recibe el MISMO espectrograma que ya usa el
% backbone CNN+Transformer, para que la comparación sea sobre el tipo de
% arquitectura, no sobre el tipo de entrada.
%
% Por qué TCN y no otra alternativa: es la arquitectura con precedente
% MÁS DIRECTO y verificable en la literatura de EMG-HGR específicamente
% para robustez INTER-SESIÓN (no solo accuracy general) -- Zanghieri et
% al. (2020, IEEE TETC; extensión 2023) evalúan TCN explícitamente en
% NinaPro DB6, el benchmark diseñado para variabilidad entre sesiones (10
% sesiones/sujeto, días distintos), mostrando mejoras sobre el estado del
% arte previo en ese escenario específico. Ver BIBLIOGRAFIA.md para la
% entrada completa.
%
% Diseño de la entrada: cada frame de espectrograma (13 frecuencias x 24
% pasos STFT x 8 canales EMG) se reorganiza como una secuencia de
% longitud 24 (eje temporal de la ventana STFT) con 13*8=104 "canales" de
% entrada por paso (frecuencia x canal EMG aplanados) -- exactamente la
% misma información que recibe el CNN Inception, solo reorganizada para
% que un TCN pueda operar sobre el eje temporal explícitamente.
%
% Arquitectura TCN (bloques residuales estándar, Bai et al. 2018):
% 4 bloques con dilatación 1,2,4,8 (campo receptivo cubre los 24 pasos:
% 1+2*(3-1)*(1+2+4+8) = 1+2*2*15 = 61 > 24), 96 canales por bloque
% (elegido para que el conteo total de parámetros sea del mismo orden que
% el backbone CNN+Transformer grande, ~3M -- ver conteo real al final del
% entrenamiento). Cada bloque: conv1D causal dilatada -> ReLU -> dropout
% -> conv1D causal dilatada -> ReLU -> dropout -> suma residual (con
% conv1x1 de ajuste de canales si es necesario) -> ReLU.
% #################################################################

function trainBackboneTCN(soloVerificar)

if nargin < 1
    soloVerificar = false;
end

clearvars -except soloVerificar; clc;
rng(9);

numTcnChannels = 96;
numTcnBlocks = 4;
dilations = 2 .^ (0:(numTcnBlocks - 1)); % 1,2,4,8
kernelSize = 3;

fprintf('=== Backbone TCN ===\n');
fprintf('  Bloques: %d, dilataciones: %s\n', numTcnBlocks, mat2str(dilations));
fprintf('  Canales por bloque: %d, kernel: %d\n', numTcnChannels, kernelSize);

if soloVerificar
    inputChannelsTest = 104; % 13 frecuencias x 8 canales EMG
    seqLenTest = 24;
    numClassesTest = 6;
    classNamesTest = categorical(1:numClassesTest);
    classWeightsTest = ones(1, numClassesTest);
    lgraph = setTCNArchitecture(inputChannelsTest, numClassesTest, classNamesTest, classWeightsTest, ...
        numTcnChannels, numTcnBlocks, dilations, kernelSize);
    analyzeNetwork(lgraph);
    fprintf('[soloVerificar] TCN: grafo válido (verificado con analyzeNetwork).\n');
    return;
end

%% SET DATASTORES PATHS
dataDirTraining = fullfile('DatastoresTran', 'training');
dataDirValidation = fullfile('DatastoresTran', 'validation');
if Shared.includeTesting
    dataDirTesting = fullfile('DatastoresTran', 'test');
end

withNoGesture = true;
classes = Shared.setNoGestureUse(withNoGesture);

trainingDatastore = SpectrogramDatastore(dataDirTraining);
validationDatastore = SpectrogramDatastore(dataDirValidation);
if Shared.includeTesting
    testingDatastore = SpectrogramDatastore(dataDirTesting);
end

clear dataDirTraining dataDirValidation withNoGesture

frameDims = trainingDatastore.FrameDimensions; % [13, 24, 8]
numFreq = frameDims(1);
seqLen = frameDims(2);
numChannelsEMG = frameDims(3);
inputChannels = numFreq * numChannelsEMG; % 104

trainingDatastore = setDataAmount(trainingDatastore, 1);
validationDatastore = setDataAmount(validationDatastore, 1);
if Shared.includeTesting
    testingDatastore = setDataAmount(testingDatastore, 1);
end

numTrainingSamples = ['Training samples: ', num2str(trainingDatastore.NumObservations)];
numValidationSamples = ['Validation samples: ', num2str(validationDatastore.NumObservations)];
if Shared.includeTesting
    numTestingSamples = ['Testing samples: ', num2str(testingDatastore.NumObservations)];
    fprintf('\n%s\n%s\n%s\n', numTrainingSamples, numValidationSamples, numTestingSamples);
else
    fprintf('\n%s\n%s\n', numTrainingSamples, numValidationSamples);
end
clear numTrainingSamples numValidationSamples numTestingSamples

classNames = Shared.setNoGestureUse(true);
classWeights = computeClassWeights(trainingDatastore, classNames);
fprintf('\nPesos de clase (compensacion de desbalance por frame):\n');
for i = 1:length(classNames)
    fprintf('  %-10s peso=%.4f\n', classNames(i), classWeights(i));
end

numClasses = trainingDatastore.NumClasses;
lgraph = setTCNArchitecture(inputChannels, numClasses, classNames, classWeights, ...
    numTcnChannels, numTcnBlocks, dilations, kernelSize);

%% NOTA IMPORTANTE: reorganización del frame [13,24,8] -> secuencia [104 x 24]
% El datastore ya entrega sequenceFoldingLayer sobre el frame completo
% (imagen 2D+canales), pero un TCN necesita una secuencia explícita. Se
% inserta una capa de reshape/permute custom (functionLayer) que convierte
% cada frame plegado (13x24x8xminibatch) a (104 x 24 x minibatch), lista
% para sequenceUnfoldingLayer + conv1D. Ver setTCNArchitecture.

miniBatchSize = 64;
maxEpochs = 10;

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.0005, ...
    'L2Regularization', 0.005, ...
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

net = trainNetwork(trainingDatastore, lgraph, options);
clear options lgraph

numParams = countLearnableParametersTCN(net);
fprintf('\nParámetros entrenables (TCN): %d\n', numParams);

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

calculateConfusionMatrix(flattenLabelsTraining, 'training-tcn');
calculateConfusionMatrix(flattenLabelsValidation, 'validation-tcn');
if Shared.includeTesting
    calculateConfusionMatrix(flattenLabelsTesting, 'testing-tcn');
end

if ~exist('Models', 'dir')
    mkdir('Models');
end
nombreModelo = sprintf('Models/Backbone_TCN_%s.mat', datestr(now,'dd-mm-yyyy_HH-MM-ss'));
save(nombreModelo, 'net', 'numParams', 'numTcnChannels', 'numTcnBlocks', 'dilations', 'kernelSize');
fprintf('\nModelo guardado en: %s\n', nombreModelo);

if Shared.includeTesting
    numMeses = 6;
    accPerMonth = nan(numMeses, 1);
    fprintf('\n--- Desempeño a lo largo del tiempo (train=Mes0, TCN) ---\n');
    for mes = 1:numMeses
        dsMes = filterByMonth(testingDatastore, mes);
        if dsMes.NumObservations == 0
            fprintf('Mes%d: sin datos, se omite.\n', mes);
            continue;
        end
        [accMes, flattenLabelsMes] = calculateAccuracy(net, dsMes);
        accPerMonth(mes) = accMes;
        fprintf('Mes%d accuracy: %.4f (n=%d secuencias)\n', mes, accMes, dsMes.NumObservations);
        calculateConfusionMatrix(flattenLabelsMes, sprintf('test-Mes%d-tcn', mes));
    end

    figure('Name', 'Desempeño a lo largo del tiempo - TCN');
    plot(1:numMeses, accPerMonth, '-o', 'LineWidth', 1.5);
    xlabel('Mes'); ylabel('Accuracy');
    title('Desempeño del modelo TCN por mes (train=Mes0, test=Mes1..Mes6)');
    xticks(1:numMeses);
    grid on;

    clear numMeses mes dsMes accMes flattenLabelsMes accPerMonth
end

end


%% ======================================================
%% FUNCTION TO ESTABLISH THE TCN ARCHITECTURE
%% ======================================================
function lgraph = setTCNArchitecture(inputChannels, numClasses, classNames, classWeights, numTcnChannels, numTcnBlocks, dilations, kernelSize)
    % IMPORTANTE (encontrado por prueba real de entrenamiento, no
    % supuesto -- ver test_smoke_tcn_training.m): el patrón
    % seqfold->(procesamiento)->sequnfold de este pipeline exige que TODO
    % el procesamiento temporal termine ANTES de sequnfold, porque
    % sequenceUnfoldingLayer requiere que su entrada tenga secuencia de
    % longitud 1 (igual que el CNN Inception, cuya salida convolucional ya
    % es una "imagen" sin eje temporal remanente, nunca una secuencia). Por
    % eso el pooling temporal global del TCN (que colapsa los 24 pasos STFT
    % a un solo vector) va DENTRO del tramo plegado, antes de sequnfold.

    lgraph = layerGraph();

    %% INPUT + FOLD (igual que el backbone CNN+Transformer, para reusar el mismo datastore)
    tempLayers = [
        sequenceInputLayer([13 24 8], "Name", "sequence", "MinLength", 1)
        sequenceFoldingLayer("Name", "seqfold")];
    lgraph = addLayers(lgraph, tempLayers);

    %% RESHAPE: [13,24,8,batch] (frame plegado) -> secuencia [104 x 24] por frame
    % Cada frame del espectrograma (13 frecuencias x 24 pasos STFT x 8
    % canales EMG) se reorganiza a una secuencia de longitud 24 con 104
    % canales por paso (13 frecuencias x 8 canales EMG aplanados) --
    % misma información que ya usa el CNN Inception, reorganizada para
    % que el eje temporal (24 pasos STFT) sea explícito para el TCN.
    tempLayers = functionLayer(@(X) reshapeSpectrogramToSequence(X), ...
        "Name", "reshape_to_sequence", "Formattable", true);
    lgraph = addLayers(lgraph, tempLayers);

    %% PROYECCIÓN INICIAL: 104 canales -> numTcnChannels (conv1x1)
    tempLayers = convolution1dLayer(1, numTcnChannels, "Name", "tcn_input_proj");
    lgraph = addLayers(lgraph, tempLayers);

    prevLayer = 'tcn_input_proj';

    %% BLOQUES TCN RESIDUALES (Bai, Kolter & Koutník 2018)
    for b = 1:numTcnBlocks
        d = dilations(b);

        blockConv1Name = sprintf('tcn%d_conv1', b);
        blockRelu1Name = sprintf('tcn%d_relu1', b);
        blockDrop1Name = sprintf('tcn%d_drop1', b);
        blockConv2Name = sprintf('tcn%d_conv2', b);
        blockRelu2Name = sprintf('tcn%d_relu2', b);
        blockDrop2Name = sprintf('tcn%d_drop2', b);
        blockAddName = sprintf('tcn%d_add', b);
        blockOutReluName = sprintf('tcn%d_outrelu', b);

        tempLayers = [
            convolution1dLayer(kernelSize, numTcnChannels, "Name", blockConv1Name, ...
                "DilationFactor", d, "Padding", "causal")
            reluLayer("Name", blockRelu1Name)
            dropoutLayer(0.3, "Name", blockDrop1Name)
            convolution1dLayer(kernelSize, numTcnChannels, "Name", blockConv2Name, ...
                "DilationFactor", d, "Padding", "causal")
            reluLayer("Name", blockRelu2Name)
            dropoutLayer(0.3, "Name", blockDrop2Name)];
        lgraph = addLayers(lgraph, tempLayers);

        tempLayers = [
            additionLayer(2, "Name", blockAddName)
            reluLayer("Name", blockOutReluName)];
        lgraph = addLayers(lgraph, tempLayers);

        lgraph = connectLayers(lgraph, prevLayer, blockConv1Name);
        lgraph = connectLayers(lgraph, blockDrop2Name, [blockAddName '/in1']);
        lgraph = connectLayers(lgraph, prevLayer, [blockAddName '/in2']); % skip residual (mismo numTcnChannels en todos los bloques, no requiere proyección)

        prevLayer = blockOutReluName;
    end

    %% Pooling temporal global (colapsa los 24 pasos a 1 vector por canal)
    %% DENTRO del tramo plegado -- salida sin eje temporal remanente.
    tempLayers = globalAveragePooling1dLayer("Name", "gap_temporal");
    lgraph = addLayers(lgraph, tempLayers);

    tempLayers = sequenceUnfoldingLayer("Name", "sequnfold");
    lgraph = addLayers(lgraph, tempLayers);

    tempLayers = flattenLayer("Name", "flatten_batch_time");
    lgraph = addLayers(lgraph, tempLayers);

    %% PROYECCIÓN A 128-D (capa congelable, análoga a dropout_2 del backbone CNN+Transformer)
    tempLayers = [
        fullyConnectedLayer(128, "Name", "fc_projection")
        reluLayer("Name", "relu_projection")
        dropoutLayer(0.4, "Name", "dropout_2")]; % mismo nombre "dropout_2" que el otro backbone, para reusar buildFeatureCache.m sin cambios
    lgraph = addLayers(lgraph, tempLayers);

    %% CABEZA: FC + softmax
    tempLayers = [
        fullyConnectedLayer(numClasses, "Name", "fc_final")
        softmaxLayer("Name", "softmax")
        classificationLayer("Name", "classoutput", "Classes", classNames, "ClassWeights", classWeights)];
    lgraph = addLayers(lgraph, tempLayers);

    %% CONEXIONES
    lgraph = connectLayers(lgraph, "seqfold/out", "reshape_to_sequence");
    lgraph = connectLayers(lgraph, "reshape_to_sequence", "tcn_input_proj");
    lgraph = connectLayers(lgraph, prevLayer, "gap_temporal");
    lgraph = connectLayers(lgraph, "seqfold/miniBatchSize", "sequnfold/miniBatchSize");
    lgraph = connectLayers(lgraph, "gap_temporal", "sequnfold/in");
    lgraph = connectLayers(lgraph, "sequnfold/out", "flatten_batch_time");
    lgraph = connectLayers(lgraph, "flatten_batch_time", "fc_projection");
    lgraph = connectLayers(lgraph, "dropout_2", "fc_final");

end


%% ======================================================
%% FUNCTION: reorganiza el frame plegado [13,24,8,N] a secuencia [104,24,N]
%% (formato "CTB": canales x tiempo x batch, requerido por convolution1dLayer)
%% ======================================================
function Y = reshapeSpectrogramToSequence(X)
    % X llega en formato dlarray con dims "SSCB" (13 x 24 x 8 x N), salida
    % de seqfold. Se permuta a [13(frecuencia), 8(canalEMG), 24(tiempo), N]
    % y se aplana frecuencia+canalEMG en un solo eje de 104 canales,
    % dejando 24 como el eje temporal explícito -- formato "CTB".
    %
    % Se usan size(Xu, k) explícitos en vez de sz(k) sobre size(Xu) --
    % confirmado por error real en tiempo de ejecución (ver
    % test_smoke_tcn_training.m) que MATLAB puede omitir dimensiones
    % singleton finales al convertir el dlarray, por lo que indexar sz(4)
    % directo puede fallar con "index exceeds array elements".
    Xu = stripdims(X); % [13, 24, 8, N]
    numFreq = size(Xu, 1);
    seqLen = size(Xu, 2);
    numCh = size(Xu, 3);
    batchSize = size(Xu, 4);
    Xp = permute(Xu, [1 3 2 4]); % [13, 8, 24, N]
    Y = reshape(Xp, [numFreq * numCh, seqLen, batchSize]); % [104, 24, N]
    Y = dlarray(Y, "CTB");
end


%% ======================================================
%% FUNCTION TO COUNT LEARNABLE PARAMETERS (post-entrenamiento, exacto)
%% ======================================================
function total = countLearnableParametersTCN(net)
    total = 0;
    for i = 1:numel(net.Layers)
        layer = net.Layers(i);
        props = properties(layer);
        for p = 1:numel(props)
            name = props{p};
            if any(strcmp(name, {'Weights', 'Bias'}))
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
        labelsPred = classify(net, sequence);
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
        sortClasses(matrixChart, classes);
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
