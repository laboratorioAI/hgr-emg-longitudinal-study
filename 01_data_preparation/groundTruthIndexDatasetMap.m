%% ======================================================
%  Script: Agregar groundTruthIndex a los datos EMG
% ======================================================
% NOTE (public repo): file paths below are placeholders. Point them at
% your own local copy of the raw dataset and its manual-segmentation
% index files (dataset itself is not distributed in this repository --
% see the top-level README for how to request it).
%
% --- Rutas base ---
rutaDatos = '<RUTA_DATASET_CRUDO>\sin_video_solo_emg\Mes4';
rutaIndices = '<RUTA_SEGMENTACION_MANUAL>\Mes4\<CARPETA_INDICES>\matFiles';
archivoIndices = fullfile(rutaIndices, 'indicesTodos.mat');

% --- Configuración ---
gestos = {'fist', 'open', 'pinch', 'waveOut', 'waveIn'};   % Gestos a procesar
rangoUsuarios = 1:29;                 % Rango de usuarios a procesar
guardarComoCopia = false;              % true = guarda "_conGT.mat", false = sobrescribe el original

% --- Almacén de avisos ---
avisos = {};

% --- Cargar índices ---
datosIndices = load(archivoIndices);
nombreVarPrincipal = fieldnames(datosIndices);
nombreVarPrincipal = nombreVarPrincipal{1};   % Ajusta el índice si hay más de una variable
indicesTodos = datosIndices.(nombreVarPrincipal);

% --- Recorrer usuarios ---
for u = rangoUsuarios
    nombreUsuario = sprintf('user%d', u);

    % Verificar que el usuario exista en indicesTodos
    if ~isfield(indicesTodos, nombreUsuario)
        msg = sprintf('[AVISO] %s no existe en indicesTodos. Se omite.', nombreUsuario);
        fprintf('%s\n', msg);
        avisos{end+1} = msg; %#ok<*SAGROW>
        continue;
    end

    carpetaUsuario = fullfile(rutaDatos, nombreUsuario);

    % --- Recorrer gestos ---
    for g = 1:numel(gestos)
        gesto = gestos{g};
        archivoGesto = fullfile(carpetaUsuario, [gesto '.mat']);

        % Verificar que el archivo exista
        if ~isfile(archivoGesto)
            msg = sprintf('[AVISO] No existe %s. Se omite.', archivoGesto);
            fprintf('%s\n', msg);
            avisos{end+1} = msg;
            continue;
        end

        % Verificar que existan los índices para ese gesto
        if ~isfield(indicesTodos.(nombreUsuario), gesto)
            msg = sprintf('[AVISO] %s no tiene índices para "%s". Se omite.', nombreUsuario, gesto);
            fprintf('%s\n', msg);
            avisos{end+1} = msg;
            continue;
        end

        % --- Cargar datos ---
        datos = load(archivoGesto);
        reps = datos.reps;
        indices = indicesTodos.(nombreUsuario).(gesto);   % 50x2

        numRepeticiones = numel(reps.(gesto).data);
        numIndices = size(indices, 1);

        if numRepeticiones ~= numIndices
            msg = sprintf('[AVISO] %s - %s: repeticiones (%d) != índices (%d). Se procesa el mínimo.', ...
                nombreUsuario, gesto, numRepeticiones, numIndices);
            fprintf('%s\n', msg);
            avisos{end+1} = msg;
        end

        n = min(numRepeticiones, numIndices);

        % --- Agregar groundTruthIndex a cada repetición ---
        for i = 1:n
            inicio = int32(indices(i, 1));
            fin    = int32(indices(i, 2));
            reps.(gesto).data{i}.groundTruthIndex = [inicio, fin];
        end

        % --- Guardar de vuelta en la estructura datos ---
        datos.reps = reps;

        % --- Guardar archivo ---
        if guardarComoCopia
            [carpeta, nombre, ext] = fileparts(archivoGesto);
            archivoSalida = fullfile(carpeta, [nombre '_conGT' ext]);
        else
            archivoSalida = archivoGesto;
        end

        save(archivoSalida, '-struct', 'datos');
        fprintf('[OK] %s - %s procesado y guardado en: %s\n', nombreUsuario, gesto, archivoSalida);
    end
end

fprintf('\nProceso completado.\n');

% --- Reimprimir todos los avisos al final ---
fprintf('\n======================================================\n');
if isempty(avisos)
    fprintf('No se generaron avisos durante el proceso.\n');
else
    fprintf('RESUMEN DE AVISOS (%d en total):\n', numel(avisos));
    fprintf('======================================================\n');
    for i = 1:numel(avisos)
        fprintf('%s\n', avisos{i});
    end
end
fprintf('======================================================\n');
