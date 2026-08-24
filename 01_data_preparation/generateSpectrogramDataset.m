% generateSpectrogramDataset - Genera los datastores de espectrogramas (CNN + Transformer) CON GROUND TRUTH
% Mes0 -> training/validation (80/20 POR REPETICION) | Mes1-6 -> test
% Genera las secuencias temporales en DatastoresTran/<split>/<gesto>/*.mat
%
% El 80/20 de Mes0 se sortea a nivel de REPETICION individual, no de
% usuario: los 19 usuarios aportan muestras tanto a training como a
% validation. Esto mantiene el sujeto controlado, de modo que la unica
% variable que cambia entre Mes0(val) y Mes1..6 es el tiempo.
clear; clc;

rng(9); % semilla fija: el split train/validation debe ser reproducible

%% CONFIGURACIÓN DE DIRECTORIOS
% NOTE (public repo): point these at your own local copy of the raw
% dataset (see top-level README -- the dataset is not distributed in
% this repository but is available from the authors on request).
dataDir = '<RUTA_DATASET_CRUDO>\sin_video_solo_emg';
outDir = '<RUTA_SALIDA>\DatastoresTran';
categories = {'fist', 'open', 'pinch', 'waveIn', 'waveOut', 'noGesture'};
splits = {'training', 'validation', 'test'};

% Crear carpetas
for s = 1:length(splits)
    for c = 1:length(categories)
        folderPath = fullfile(outDir, splits{s}, categories{c});
        if ~exist(folderPath, 'dir')
            mkdir(folderPath);
        end
    end
end

%% --- CONTADORES DE TRACKING ---
cont.archivosGuardados       = 0;
cont.repeticionesSinGT       = 0;
cont.repeticionesConGT       = 0;
cont.porSplit = struct('training', 0, 'validation', 0, 'test', 0);
avisos = {};

allowedUsers = {'user4', 'user5', 'user6', 'user8', 'user9', 'user12', ...
            'user14', 'user21', 'user23', 'user24', 'user26', 'user35', ...
            'user37', 'user40', 'user42', 'user43', 'user49', 'user55', 'user57'};

%% GENERACIÓN DE SECUENCIAS
meses = dir(fullfile(dataDir, 'Mes*'));

for i = 1:length(meses)
    mesName = meses(i).name;

    % --- División académica a nivel de mes ---
    if strcmp(mesName, 'Mes0')
        tipoMes = 'trainval';   % se decide 80/20 por usuario más abajo
    elseif ismember(mesName, {'Mes1', 'Mes2', 'Mes3', 'Mes4', 'Mes5', 'Mes6'})
        tipoMes = 'test';
    else
        continue;
    end

    fprintf('Procesando %s -> %s\n', mesName, upper(tipoMes));

    users = dir(fullfile(dataDir, mesName, 'user*'));
    for j = 1:length(users)

        userName = users(j).name;

        % Filtro: solo usuarios permitidos (aplica tanto a Mes0 como a Mes1-6)
        if ~ismember(userName, allowedUsers)
            continue;
        end

        % --- Asignar split según el tipo de mes ---
        % CORREGIDO (2026-07-28): antes el sorteo 80/20 estaba AQUI, en el
        % bucle de usuarios, de modo que cada usuario entero caia en
        % training o en validation. Eso hacia que training y validation
        % quedaran disjuntos por sujeto (10 vs 9 usuarios, interseccion
        % vacia), mientras que test contenia los 19. Consecuencia:
        % Mes0(val) media generalizacion INTER-SUJETO (sujetos nunca
        % vistos) y Mes1-6 media una mezcla donde 10 de 19 usuarios si
        % habian sido entrenados. La brecha sujeto-visto vs sujeto-nuevo
        % medida fue de ~2.7 puntos (0.884 vs 0.857), suficiente para
        % invertir el orden aparente y producir el artefacto de "Mes0 como
        % valle en vez de pico" observado en los 9 modelos.
        % Ahora el sorteo se hace POR REPETICION (mas abajo, en el bucle r),
        % de forma que los 19 usuarios estan presentes tanto en training
        % como en validation. Asi la unica variable que cambia entre
        % Mes0(val) y Mes1..6 es el TIEMPO, con el sujeto controlado.
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
                % 1. Extraer el campo dinámico (ej. 'fist', 'open')
                campos = fieldnames(matData.reps(1));
                campoGesto = campos{1};
                estructuraGesto = matData.reps(1).(campoGesto);

                % 2. Obtener el nombre del gesto
                gesture = estructuraGesto.gestureName;
                if strcmpi(gesture, 'relax') || contains(lower(gesture), 'relax')
                    gesture = 'noGesture';
                end

                % 3. Extraer la celda con las repeticiones
                celdasRepeticiones = estructuraGesto.data;

                for r = 1:length(celdasRepeticiones)

                    % --- EXTRACCIÓN CONTRA INCONSISTENCIAS ---
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
                    % --------------------------------------------------

                    % --- SPLIT POR REPETICION (solo Mes0) ---
                    % El sorteo 80/20 se hace aqui, a nivel de repeticion
                    % individual, para que cada usuario aporte muestras
                    % tanto a training como a validation. Ver la nota
                    % extensa en el bucle de usuarios sobre por que el
                    % sorteo por usuario invalidaba la comparacion
                    % temporal.
                    if strcmp(tipoMes, 'trainval')
                        if rand < 0.8
                            splitName = 'training';
                        else
                            splitName = 'validation';
                        end
                    end

                    % --- OBTENER groundTruthIndex SI EXISTE ---
                    tieneGT = isstruct(repContent) && isfield(repContent, 'groundTruthIndex') ...
                              && ~strcmp(gesture, 'noGesture');

                    if tieneGT
                        gtIdx = double(repContent.groundTruthIndex);   % [inicio fin]
                        cont.repeticionesConGT = cont.repeticionesConGT + 1;
                    else
                        cont.repeticionesSinGT = cont.repeticionesSinGT + 1;
                        msg = sprintf('[AVISO] %s (%s) - rep %d: sin groundTruthIndex. Se procesa sin GT.', fileName, mesName, r);
                        avisos{end+1} = msg; %#ok<*SAGROW>
                    end

                    signal = Shared.preprocessSignal(signal);
                    numMuestras = size(signal, 1);

                    % --- CONSTRUIR VECTOR groundTruth BINARIO ---
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

                            frameSignal = signal(inicio:finish, :);
                            spectrogram = Shared.generateSpectrograms(frameSignal);

                            sequenceData{w, 1} = spectrogram;
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
        if strcmp(tipoMes, 'trainval')
            fprintf('  - %s (%s) procesado -> training/validation (80/20 por repeticion)\n', userName, mesName);
        else
            fprintf('  - %s (%s) procesado -> test\n', userName, mesName);
        end
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

fprintf('\n¡Generación LSTM completada en MATLAB!\n');