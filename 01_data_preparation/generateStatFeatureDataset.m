% #################################################################
% Experimento 2A (Tarea 2 del plan de respuesta a revisores, 2026-07-23):
% RED SIN CNN -- features estadísticos clásicos por ventana y canal EMG,
% en vez de espectrograma + CNN Inception. Mismo patrón que el modelo
% ANN-DTW de Benalcázar et al. (2018, EUSIPCO) y el SVM de Barona López et
% al. (2020, Sensors) -- ver BIBLIOGRAFIA.md.
%
% Reusa EXACTAMENTE el mismo ventaneo/etiquetado/split que
% CNN-Transformer/generateSpectrogramDataset.m (Shared.FRAME_WINDOW,
% Shared.WINDOW_STEP, la regla de tolerancia de gtWindows, el mismo
% Shared.preprocessSignal, los mismos allowedUsers, la misma partición
% Mes0->training/validation 80/20 y Mes1-6->test) -- la ÚNICA diferencia
% deliberada es qué se calcula por ventana: en vez de un espectrograma STFT
% (imagen tiempo-frecuencia), se calculan 4 estadísticos por canal:
%   - MAV  (mean absolute value)
%   - RMS  (root mean square)
%   - SD   (desviación estándar)
%   - ENERGIA (suma de cuadrados)
% NOTA: NO se incluye "envolvente de Hilbert" como feature adicional
% porque la señal que llega aquí (Shared.preprocessSignal) YA es una
% envolvente lineal (rectificada con abs() + pasabajos Butterworth) -- es
% exactamente la técnica clásica de "envolvente lineal" de EMG. Agregar
% Hilbert encima sería una feature redundante con el preprocesamiento ya
% aplicado, no una fuente de información nueva.
% Con 8 canales x 4 estadísticos = 32 features por ventana.
%
% Salida: DatastoresStatFeatures/<split>/<gesto>/*.mat, con estructura
% 'data.sequenceData' (Nx3: {featureVector, label, timestamp}) y
% 'data.groundTruth' -- MISMA estructura que usa SpectrogramDatastore.m,
% para poder reusar esa misma clase (readFile de Shared.m ya funciona
% igual porque no distingue el contenido de sequenceData{i,1} más que
% para pasarlo como "frame" -- ver StatFeatureDatastore.m, que hereda el
% mismo patrón pero sin asumir 2D/imagen).
% #################################################################

clear; clc;

%% CONFIGURACIÓN DE DIRECTORIOS
% NOTE (public repo): point these at your own local copy of the raw
% dataset (see top-level README -- the dataset is not distributed in
% this repository but is available from the authors on request).
dataDir = '<RUTA_DATASET_CRUDO>\sin_video_solo_emg';
outDir = '<RUTA_SALIDA>\DatastoresStatFeatures';
categories = {'fist', 'open', 'pinch', 'waveIn', 'waveOut', 'noGesture'};
splits = {'training', 'validation', 'test'};

for s = 1:length(splits)
    for c = 1:length(categories)
        folderPath = fullfile(outDir, splits{s}, categories{c});
        if ~exist(folderPath, 'dir')
            mkdir(folderPath);
        end
    end
end

%% --- CONTADORES DE TRACKING (idénticos a generateSpectrogramDataset.m) ---
cont.archivosGuardados       = 0;
cont.repeticionesSinGT       = 0;
cont.repeticionesConGT       = 0;
cont.porSplit = struct('training', 0, 'validation', 0, 'test', 0);
avisos = {};

allowedUsers = {'user4', 'user5', 'user6', 'user8', 'user9', 'user12', ...
            'user14', 'user21', 'user23', 'user24', 'user26', 'user35', ...
            'user37', 'user40', 'user42', 'user43', 'user49', 'user55', 'user57'};

rng(9); % misma semilla que generateSpectrogramDataset.m para el split 80/20 de Mes0

%% GENERACIÓN DE SECUENCIAS
meses = dir(fullfile(dataDir, 'Mes*'));

for i = 1:length(meses)
    mesName = meses(i).name;

    if strcmp(mesName, 'Mes0')
        tipoMes = 'trainval';
    elseif ismember(mesName, {'Mes1', 'Mes2', 'Mes3', 'Mes4', 'Mes5', 'Mes6'})
        tipoMes = 'test';
    else
        continue;
    end

    fprintf('Procesando %s -> %s\n', mesName, upper(tipoMes));

    users = dir(fullfile(dataDir, mesName, 'user*'));
    for j = 1:length(users)
        userName = users(j).name;

        if ~ismember(userName, allowedUsers)
            continue;
        end

        % CORREGIDO (2026-07-28): mismo bug que generateSpectrogramDataset.m
        % -- el sorteo 80/20 estaba aqui, por usuario completo, dejando
        % training/validation disjuntos por sujeto. Ahora se sortea por
        % repeticion (mas abajo, bucle r), para que los 19 usuarios
        % aporten muestras a ambos splits y Mes0(val) mida lo mismo que
        % Mes1..6 (mismos sujetos, solo cambia el tiempo).
        if ~strcmp(tipoMes, 'trainval')
            splitName = 'test';
        end

        matFiles = dir(fullfile(dataDir, mesName, userName, '*.mat'));

        for k = 1:length(matFiles)
            fileName = matFiles(k).name;
            if strcmp(fileName, 'userData.mat')
                continue;
            end

            filePath = fullfile(matFiles(k).folder, fileName);
            matData = load(filePath);

            if isfield(matData, 'reps')
                campos = fieldnames(matData.reps(1));
                campoGesto = campos{1};
                estructuraGesto = matData.reps(1).(campoGesto);

                gesture = estructuraGesto.gestureName;
                if strcmpi(gesture, 'relax') || contains(lower(gesture), 'relax')
                    gesture = 'noGesture';
                end

                celdasRepeticiones = estructuraGesto.data;

                for r = 1:length(celdasRepeticiones)
                    repContent = celdasRepeticiones{r};
                    if isempty(repContent)
                        continue;
                    elseif isstruct(repContent) && isfield(repContent, 'emg')
                        signal = repContent.emg;
                    elseif isnumeric(repContent)
                        signal = repContent;
                    else
                        continue;
                    end

                    % --- SPLIT POR REPETICION (solo Mes0), ver nota arriba ---
                    if strcmp(tipoMes, 'trainval')
                        if rand < 0.8
                            splitName = 'training';
                        else
                            splitName = 'validation';
                        end
                    end

                    tieneGT = isstruct(repContent) && isfield(repContent, 'groundTruthIndex') ...
                              && ~strcmp(gesture, 'noGesture');

                    if tieneGT
                        gtIdx = double(repContent.groundTruthIndex);
                        cont.repeticionesConGT = cont.repeticionesConGT + 1;
                    else
                        cont.repeticionesSinGT = cont.repeticionesSinGT + 1;
                        msg = sprintf('[AVISO] %s (%s) - rep %d: sin groundTruthIndex. Se procesa sin GT.', fileName, mesName, r);
                        avisos{end+1} = msg; %#ok<*SAGROW>
                    end

                    signal = Shared.preprocessSignal(signal);
                    numMuestras = size(signal, 1);

                    groundTruth = zeros(numMuestras, 1);
                    if tieneGT
                        inicioGT = max(1, gtIdx(1));
                        finGT    = min(numMuestras, gtIdx(2));
                        if finGT >= inicioGT
                            groundTruth(inicioGT:finGT) = 1;
                        end
                    end
                    numGesturePoints = sum(groundTruth == 1);

                    numWindows = floor((length(signal) - Shared.FRAME_WINDOW) / Shared.WINDOW_STEP) + 1;

                    if numWindows > 0
                        sequenceData = cell(numWindows, 3);
                        gtWindows = zeros(numWindows, 1);

                        for w = 1:numWindows
                            inicio = 1 + (w-1) * Shared.WINDOW_STEP;
                            finish = inicio + Shared.FRAME_WINDOW - 1;
                            timestamp = inicio + floor(Shared.FRAME_WINDOW / 2);

                            frameSignal = signal(inicio:finish, :); % FRAME_WINDOW x 8 canales
                            featVec = computeClassicFeatures(frameSignal); % 32x1 (8 canales x 4 stats)

                            sequenceData{w, 1} = featVec;
                            sequenceData{w, 3} = timestamp;

                            if tieneGT
                                frameGT = groundTruth(inicio:finish);
                                totalOnes = sum(frameGT == 1);

                                esGesto = totalOnes >= Shared.FRAME_WINDOW * Shared.TOLERANCE_WINDOW || ...
                                    (numGesturePoints > 0 && totalOnes >= numGesturePoints * Shared.TOLERNCE_GESTURE);

                                if esGesto
                                    sequenceData{w, 2} = gesture;
                                    gtWindows(w) = 1;
                                else
                                    sequenceData{w, 2} = 'noGesture';
                                end
                            else
                                sequenceData{w, 2} = gesture;
                            end
                        end

                        data.sequenceData = sequenceData;
                        if ~strcmp(gesture, 'noGesture')
                            data.groundTruth = transpose(gtWindows);
                        end

                        cleanFileName = strrep(fileName, '.mat', '');
                        saveName = sprintf('%s_%s_%s_rep%d_seq.mat', mesName, userName, cleanFileName, r);
                        save(fullfile(outDir, splitName, gesture, saveName), 'data');
                        cont.archivosGuardados = cont.archivosGuardados + 1;
                        cont.porSplit.(splitName) = cont.porSplit.(splitName) + 1;

                        clear data
                    end
                end
            end
        end
        fprintf('  - %s (%s) procesado -> %s\n', userName, mesName, splitName);
    end
end

%% RESUMEN FINAL
fprintf('\n======================================================\n');
fprintf('Archivos .mat guardados:        %d\n', cont.archivosGuardados);
fprintf('  - training:   %d\n', cont.porSplit.training);
fprintf('  - validation: %d\n', cont.porSplit.validation);
fprintf('  - test:       %d\n', cont.porSplit.test);
fprintf('Repeticiones CON groundTruthIndex: %d\n', cont.repeticionesConGT);
fprintf('Repeticiones SIN groundTruthIndex: %d\n', cont.repeticionesSinGT);
fprintf('======================================================\n');

if ~isempty(avisos)
    fprintf('\nRESUMEN DE AVISOS (%d):\n', numel(avisos));
    for a = 1:numel(avisos)
        fprintf('%s\n', avisos{a});
    end
end

fprintf('\nGeneración de dataset de features clásicos (Experimento 2A) completada.\n');


%% ======================================================
%% FUNCTION: 4 ESTADÍSTICOS CLÁSICOS POR CANAL (MAV, RMS, SD, ENERGÍA)
%% ======================================================
function featVec = computeClassicFeatures(frameSignal)
    % frameSignal: FRAME_WINDOW x numChannels (ya rectificada + pasabajos,
    % ver Shared.preprocessSignal -- es una envolvente lineal, no señal
    % cruda). Devuelve un vector columna de 4*numChannels features, en el
    % orden [MAV_ch1..8, RMS_ch1..8, SD_ch1..8, ENERGIA_ch1..8].
    mav = mean(abs(frameSignal), 1);
    rms_ = sqrt(mean(frameSignal.^2, 1));
    sd = std(frameSignal, 0, 1);
    energia = sum(frameSignal.^2, 1);
    featVec = [mav, rms_, sd, energia]';
end
