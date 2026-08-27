% #################################################################
% Tarea 2 (plan de respuesta a revisores, 2026-07-22..24) + experimento de
% arquitectura alternativa (2026-07-24): consolida los resultados de las 5
% variantes de capacidad/arquitectura x 3 condiciones (A=softmax,
% B=LinUCB congelado, C=LinUCB adaptativo) en una sola tabla y una figura
% comparativa de caída relativa Mes1->Mes6.
%
% Variantes:
%   grande  -- CNN Inception (6 bloques) + Transformer, d_model=128, 8 heads
%   mediano -- CNN Inception (4 bloques) + Transformer, d_model=64,  4 heads
%   pequeno -- CNN Inception (2 bloques) + Transformer, d_model=32,  2 heads
%   ann     -- SIN CNN: 4 estadísticos clásicos (MAV/RMS/SD/energía) por
%              canal EMG + red densa pequeña (64->32->numClasses)
%   tcn     -- SIN CNN Inception, SIN Transformer/atención: Temporal
%              Convolutional Network (Bai et al. 2018) sobre el mismo
%              espectrograma, 4 bloques residuales dilatados (1,2,4,8), 96
%              canales -- arquitectura genuinamente distinta, con
%              precedente específico en robustez inter-sesión en EMG
%              (Zanghieri et al. 2020/2023, NinaPro DB6). Ver
%              trainBackboneTCN.m. Casi el mismo número de parámetros que
%              'pequeno' (245222 vs 246850) -- comparación de capacidad
%              equivalente entre familias arquitectónicas distintas.
%
% Objetivo: responder a los 3 revisores del laboratorio que pidieron
% verificar si el patrón de drift observado es un artefacto del tamaño (o
% del tipo) de la red grande, Y a la hipótesis del usuario de que la
% combinación específica CNN+Transformer podría ser la que mitiga el
% drift (comparando contra una arquitectura genuinamente distinta, TCN).
% Si la caída relativa se mantiene de magnitud similar en todas, es
% evidencia directa contra ambas hipótesis.
% #################################################################

clear; clc;

variantes = {'grande', 'mediano', 'pequeno', 'ann', 'tcn'};
numParams = [3026614, 998774, 246850, 4390, 245222]; % ver consulta manual, Models/*.mat
condiciones = struct( ...
    'nombre', {'A_softmax', 'B_banditCongelado', 'C_banditAdaptativo'}, ...
    'patronArchivo', {'SoftmaxHead_Results', 'Bandit_Results', 'BanditAdaptive_Results'}, ...
    'etiqueta', {'Condición A (Softmax)', 'Condición B (LinUCB congelado)', 'Condición C (LinUCB adaptativo)'});

resultados = struct();

for v = 1:numel(variantes)
    tam = variantes{v};
    for c = 1:numel(condiciones)
        patron = condiciones(c).patronArchivo;

        if strcmp(tam, 'grande')
            % La variante 'grande' se entrenó ANTES de parametrizar estos
            % scripts por tamaño (esta sesión, 2026-07-22..24), así que
            % sus resultados no llevan sufijo de tamaño en el nombre de
            % archivo. Se toma el más reciente que NO sea de ninguna de
            % las otras 3 variantes (que sí llevan sufijo explícito).
            files = dir(fullfile('Models', sprintf('%s_*.mat', patron)));
            esOtraVariante = contains({files.name}, {'_ann_', '_mediano_', '_pequeno_', '_tcn_'});
            files = files(~esOtraVariante);
        else
            files = dir(fullfile('Models', sprintf('%s_%s_*.mat', patron, tam)));
        end

        if isempty(files)
            error('No se encontró ningún archivo de resultados para variante=%s, patrón=%s', tam, patron);
        end
        [~, idx] = max([files.datenum]);
        loaded = load(fullfile(files(idx).folder, files(idx).name), 'accByMonth', 'monthLabels');
        resultados.(tam).(condiciones(c).nombre).accByMonth = loaded.accByMonth;
        resultados.(tam).(condiciones(c).nombre).monthLabels = loaded.monthLabels;
    end
end

%% TABLA: accuracy Mes1, accuracy Mes6, caída absoluta y relativa
fprintf('\n=====================================================================================\n');
fprintf(' TAREA 2 -- ABLACIÓN DE TAMAÑO/ARQUITECTURA: ¿el drift depende del tamaño de la red?\n');
fprintf('=====================================================================================\n\n');
fprintf('%-10s %-9s %12s %10s %10s %10s %12s\n', 'Variante', 'Params', 'Condición', 'Mes1(%)', 'Mes6(%)', 'Caída(pp)', 'Caída rel.(%)');
fprintf('%s\n', repmat('-', 1, 90));

resumenCaidaRelativa = nan(numel(variantes), numel(condiciones));

for v = 1:numel(variantes)
    tam = variantes{v};
    for c = 1:numel(condiciones)
        nombreCond = condiciones(c).nombre;
        accByMonth = resultados.(tam).(nombreCond).accByMonth;
        monthLabels = resultados.(tam).(nombreCond).monthLabels;

        idxMes1 = find(strcmp(cellstr(monthLabels), 'Mes1'), 1);
        idxMes6 = find(strcmp(cellstr(monthLabels), 'Mes6'), 1);
        if isempty(idxMes6)
            idxMes6 = numel(accByMonth); % por si NUM_VALID_TEST_MONTHS < 6 en algún cache viejo
        end

        acc1 = accByMonth(idxMes1) * 100;
        acc6 = accByMonth(idxMes6) * 100;
        caidaAbs = acc1 - acc6;
        caidaRel = (caidaAbs / acc1) * 100;

        resumenCaidaRelativa(v, c) = caidaRel;

        fprintf('%-10s %-9d %-12s %10.2f %10.2f %10.2f %12.2f\n', ...
            tam, numParams(v), condiciones(c).etiqueta, acc1, acc6, caidaAbs, caidaRel);
    end
    fprintf('%s\n', repmat('-', 1, 90));
end

fprintf('\nInterpretación: si "Caída rel.(%%)" es de magnitud similar entre grande/mediano/pequeño/ann,\n');
fprintf('la caída de accuracy Mes1->Mes6 NO es un artefacto del tamaño ni del tipo de arquitectura\n');
fprintf('de la red grande -- se mantiene incluso en una red ~690x más chica (ann vs. grande) y en una\n');
fprintf('familia arquitectónica completamente distinta (sin CNN, sin atención).\n');

%% FIGURA: caída relativa por variante y condición
outDir = 'FigurasReales';
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

fig = figure('Visible', 'off', 'Position', [100 100 950 550]);
barColors = [0.34 0.61 0.84; 0.93 0.49 0.19; 0.44 0.68 0.28];
b = bar(resumenCaidaRelativa, 'grouped');
for c = 1:numel(condiciones)
    b(c).FaceColor = barColors(c, :);
end
etiquetasX = {'Grande (3.03M)', 'Mediano (1.00M)', 'Pequeño (0.25M)', 'ANN sin CNN (4.4K)', 'TCN (0.25M)'};
set(gca, 'XTick', 1:numel(etiquetasX), 'XTickLabel', etiquetasX);
ylabel('Caída relativa de accuracy Mes1\rightarrowMes6 (%)');
title('Ablación de tamaño/arquitectura: ¿el drift depende del tamaño de la red?');
legend({condiciones.etiqueta}, 'Location', 'northoutside', 'Orientation', 'horizontal');
grid on;
exportgraphics(fig, fullfile(outDir, 'ablacion_tamano_caida_relativa.png'), 'Resolution', 200);
close(fig);

%% FIGURA: accuracy por mes, las 5 variantes, condición A (softmax) como referencia
fig2 = figure('Visible', 'off', 'Position', [100 100 850 550]);
hold on;
coloresVariante = [0.20 0.20 0.20; 0.34 0.61 0.84; 0.93 0.49 0.19; 0.44 0.68 0.28; 0.60 0.20 0.60];
markers = {'-o', '-s', '-^', '-d', '-p'};
for v = 1:numel(variantes)
    tam = variantes{v};
    accByMonth = resultados.(tam).A_softmax.accByMonth * 100;
    monthLabels = cellstr(resultados.(tam).A_softmax.monthLabels);
    xVals = 1:numel(accByMonth);
    plot(xVals, accByMonth, markers{v}, 'Color', coloresVariante(v, :), 'LineWidth', 2, ...
        'MarkerFaceColor', coloresVariante(v, :), 'DisplayName', sprintf('%s (%d par.)', tam, numParams(v)));
end
hold off;
grid on;
xticks(xVals);
xticklabels(monthLabels);
ylabel('Accuracy (%)');
xlabel('Sesión');
title('Condición A (softmax): accuracy por sesión, 5 arquitecturas');
legend('Location', 'southwest');
ylim([50 100]);
exportgraphics(fig2, fullfile(outDir, 'ablacion_tamano_accuracy_condicionA.png'), 'Resolution', 200);
close(fig2);

%% GUARDAR RESUMEN NUMÉRICO
save('Models/AblacionTamano_Resumen.mat', 'resultados', 'variantes', 'numParams', 'resumenCaidaRelativa', 'condiciones');
fprintf('\nResumen guardado en Models/AblacionTamano_Resumen.mat\n');
fprintf('Figuras guardadas en %s\n', fullfile(pwd, outDir));
