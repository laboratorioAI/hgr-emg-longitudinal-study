%% ======================================================
%  Script: Agregar groundTruthIndex a partir de la segmentacion manual
%  de los estudiantes (Mes1, Mes5, Mes6)
%
%  Mapeo validado contra segmentacion-metodos-numericos\seg_estudiantes.jpg,
%  los indicesTodos.mat reales de cada estudiante, y sus informes en PDF.
%
%  NOTE (public repo): the original script keyed each folder by the real
%  full name of the student who performed that manual segmentation batch.
%  Those names are personal data of research collaborators (not of the
%  EMG study subjects) and have been replaced here with generic aliases
%  (Estudiante_A..G) that preserve the exact mes/usuarios/archivoIndices
%  mapping. File paths are placeholders; see the top-level README.
% ======================================================
rutaBase = '<RUTA_REPO_DATOS_CRUDOS>';
rutaDatosBase = fullfile(rutaBase, 'sin_video_solo_emg');
rutaSegBase = fullfile(rutaBase, 'segmentacion-metodos-numericos');

gestos = {'fist', 'open', 'pinch', 'waveOut', 'waveIn'};
guardarComoCopia = false;  % true = guarda "_conGT.mat" (no toca el original), false = sobrescribe

% --- Configuracion: que estudiante aporta que usuarios, para que mes ---
% archivoIndices: '' = usar la ruta por defecto <carpeta>\matFiles\indicesTodos.mat.
%                 Si se especifica, se usa esa ruta exacta (caso user43: Estudiante_G lo
%                 entrego aparte, directo en la raiz de su carpeta, no en matFiles\).
config = struct('carpeta', {}, 'mes', {}, 'usuarios', {}, 'archivoIndices', {});
config(end+1) = struct('carpeta', 'Estudiante_A', 'mes', 1, 'usuarios', [2 3 4 5 6 7 8 9 19 20 21], 'archivoIndices', '');
config(end+1) = struct('carpeta', 'Estudiante_B', 'mes', 1, 'usuarios', [22 23 24 25 26 27 28 29 30 31 32], 'archivoIndices', '');
config(end+1) = struct('carpeta', 'Estudiante_C', 'mes', 5, 'usuarios', [3 4 5 6], 'archivoIndices', '');
config(end+1) = struct('carpeta', 'Estudiante_D', 'mes', 5, 'usuarios', [7 8 9 12 13 14 15 16 18 19 20], 'archivoIndices', '');
config(end+1) = struct('carpeta', 'Estudiante_E', 'mes', 5, 'usuarios', [21 22 23 24 25 26 30 35 37 38 40], 'archivoIndices', '');
config(end+1) = struct('carpeta', 'Estudiante_F', 'mes', 5, 'usuarios', [41 42 43 49 53 55 56 57 61 62], 'archivoIndices', '');
config(end+1) = struct('carpeta', 'Estudiante_F', 'mes', 6, 'usuarios', 4, 'archivoIndices', '');
config(end+1) = struct('carpeta', 'Estudiante_H', 'mes', 6, 'usuarios', [5 6 8 9 12 14 21 23 24 26 35], 'archivoIndices', '');
config(end+1) = struct('carpeta', 'Estudiante_G', 'mes', 6, 'usuarios', [37 40 42 49 55 57], 'archivoIndices', ''); % 23,24,26,35 ya cubiertos por Estudiante_H
config(end+1) = struct('carpeta', 'Estudiante_G', 'mes', 6, 'usuarios', 43, ...
    'archivoIndices', fullfile(rutaSegBase, 'Estudiante_G', 'indicesTodos.mat')); % entregado aparte por otro estudiante, validado: 5 gestos x 50x2

% Huecos conocidos, documentados y deliberadamente NO procesados:
%   Mes1: users 36,48,50,52,54,60,64 (Estudiante_C - "no se realizaron por falta de sus archivos", confirmado en su PDF y en disco)

avisos = {};

for c = 1:numel(config)
    cfg = config(c);
    if isempty(cfg.archivoIndices)
        archivoIndices = fullfile(rutaSegBase, cfg.carpeta, 'matFiles', 'indicesTodos.mat');
    else
        archivoIndices = cfg.archivoIndices;
    end
    datosIndices = load(archivoIndices);
    nombreVarPrincipal = fieldnames(datosIndices);
    nombreVarPrincipal = nombreVarPrincipal{1};
    indicesTodos = datosIndices.(nombreVarPrincipal);

    rutaDatosMes = fullfile(rutaDatosBase, sprintf('Mes%d', cfg.mes));

    for u = cfg.usuarios
        nombreUsuario = sprintf('user%d', u);

        if ~isfield(indicesTodos, nombreUsuario)
            msg = sprintf('[AVISO] Mes%d - %s no existe en indicesTodos de %s. Se omite.', cfg.mes, nombreUsuario, cfg.carpeta);
            fprintf('%s\n', msg);
            avisos{end+1} = msg; %#ok<*SAGROW>
            continue;
        end

        carpetaUsuario = fullfile(rutaDatosMes, nombreUsuario);

        for g = 1:numel(gestos)
            gesto = gestos{g};
            archivoGesto = fullfile(carpetaUsuario, [gesto '.mat']);

            if ~isfile(archivoGesto)
                msg = sprintf('[AVISO] No existe %s. Se omite.', archivoGesto);
                fprintf('%s\n', msg);
                avisos{end+1} = msg;
                continue;
            end

            if ~isfield(indicesTodos.(nombreUsuario), gesto)
                msg = sprintf('[AVISO] Mes%d - %s no tiene indices para "%s" en %s. Se omite.', cfg.mes, nombreUsuario, gesto, cfg.carpeta);
                fprintf('%s\n', msg);
                avisos{end+1} = msg;
                continue;
            end

            datos = load(archivoGesto);
            reps = datos.reps;
            indices = indicesTodos.(nombreUsuario).(gesto);

            numRepeticiones = numel(reps.(gesto).data);
            numIndices = size(indices, 1);

            if numRepeticiones ~= numIndices
                msg = sprintf('[AVISO] Mes%d %s - %s: repeticiones (%d) != indices (%d). Se procesa el minimo.', ...
                    cfg.mes, nombreUsuario, gesto, numRepeticiones, numIndices);
                fprintf('%s\n', msg);
                avisos{end+1} = msg;
            end

            n = min(numRepeticiones, numIndices);

            for i = 1:n
                inicio = int32(indices(i, 1));
                fin    = int32(indices(i, 2));
                reps.(gesto).data{i}.groundTruthIndex = [inicio, fin];
            end

            datos.reps = reps;

            if guardarComoCopia
                [carpeta, nombre, ext] = fileparts(archivoGesto);
                archivoSalida = fullfile(carpeta, [nombre '_conGT' ext]);
            else
                archivoSalida = archivoGesto;
            end

            save(archivoSalida, '-struct', 'datos');
            fprintf('[OK] Mes%d - %s - %s procesado (fuente: %s) -> %s\n', cfg.mes, nombreUsuario, gesto, cfg.carpeta, archivoSalida);
        end
    end
end

fprintf('\nProceso completado.\n');
fprintf('\n======================================================\n');
fprintf('Huecos documentados sin segmentacion (deliberadamente no procesados):\n');
fprintf('  Mes1: users 36,48,50,52,54,60,64 (archivos crudos no existen - confirmado en PDF de Estudiante_C)\n');
fprintf('======================================================\n');
if isempty(avisos)
    fprintf('No se generaron avisos adicionales durante el proceso.\n');
else
    fprintf('RESUMEN DE AVISOS (%d en total):\n', numel(avisos));
    fprintf('======================================================\n');
    for i = 1:numel(avisos)
        fprintf('%s\n', avisos{i});
    end
end
fprintf('======================================================\n');
