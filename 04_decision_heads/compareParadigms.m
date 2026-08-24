% #################################################################
% PASO 3: COMPARACIÓN FINAL - SOFTMAX (congelado) vs. LinUCB (congelado)
% vs. LinUCB ADAPTATIVO
% Carga los resultados más recientes de trainSoftmaxHead.m, context_bandit.m
% y context_bandit_adaptive.m (mismo cache de features; softmax y LinUCB
% congelado entrenados una sola vez con Mes0 y nunca más actualizados;
% LinUCB adaptativo parte del mismo warm-start pero sigue aprendiendo mes
% a mes solo con la señal de recompensa) y grafica las tres curvas juntas,
% limitado a Mes0(baseline)..Mes{Shared.NUM_VALID_TEST_MONTHS} -- los
% meses restantes todavía no tienen anotación (groundTruthIndex)
% consistente con el resto, ver Shared.m.
% #################################################################

clear; clc;

softmaxFiles = dir(fullfile('Models', 'SoftmaxHead_Results_*.mat'));
banditFiles = dir(fullfile('Models', 'Bandit_Results_*.mat'));
adaptiveFiles = dir(fullfile('Models', 'BanditAdaptive_Results_*.mat'));
if isempty(softmaxFiles) || isempty(banditFiles) || isempty(adaptiveFiles)
    error('Faltan resultados: corre trainSoftmaxHead.m, context_bandit.m y context_bandit_adaptive.m primero.');
end

[~, idxS] = max([softmaxFiles.datenum]);
[~, idxB] = max([banditFiles.datenum]);
[~, idxA] = max([adaptiveFiles.datenum]);
softmaxFile = fullfile(softmaxFiles(idxS).folder, softmaxFiles(idxS).name);
banditFile = fullfile(banditFiles(idxB).folder, banditFiles(idxB).name);
adaptiveFile = fullfile(adaptiveFiles(idxA).folder, adaptiveFiles(idxA).name);
fprintf('Softmax          : %s\n', softmaxFile);
fprintf('LinUCB congelado : %s\n', banditFile);
fprintf('LinUCB adaptativo: %s\n', adaptiveFile);

S = load(softmaxFile, 'accByMonth', 'f1ByMonth', 'monthLabels');
B = load(banditFile, 'accByMonth', 'f1ByMonth', 'monthLabels');
Ad = load(adaptiveFile, 'accByMonth', 'f1ByMonth', 'monthLabels');

% Alinear longitudes por si algun resultado quedo de antes de fijar
% Shared.NUM_VALID_TEST_MONTHS (para no graficar con ejes desparejos).
numPuntos = min([numel(S.monthLabels), numel(B.monthLabels), numel(Ad.monthLabels)]);
if numel(S.monthLabels) ~= numel(B.monthLabels) || numel(S.monthLabels) ~= numel(Ad.monthLabels)
    warning('Los tres resultados tienen distinto numero de meses; probablemente alguno quedo desactualizado. Se recorta al minimo comun (%d puntos). Vuelve a correr los scripts que falten con el Shared.NUM_VALID_TEST_MONTHS actual.', numPuntos);
end
monthLabels = S.monthLabels(1:numPuntos);
x = 1:numPuntos;

Sacc = S.accByMonth(1:numPuntos); Sf1 = S.f1ByMonth(1:numPuntos);
Bacc = B.accByMonth(1:numPuntos); Bf1 = B.f1ByMonth(1:numPuntos);
Aacc = Ad.accByMonth(1:numPuntos); Af1 = Ad.f1ByMonth(1:numPuntos);

%% GRÁFICA DE ACCURACY
figure('Name', 'Comparación de paradigmas - Accuracy', 'Position', [100, 100, 800, 500]);
plot(x, Sacc * 100, '-o', 'LineWidth', 2, 'DisplayName', 'Softmax (congelado)');
hold on;
plot(x, Bacc * 100, '-s', 'LineWidth', 2, 'DisplayName', 'LinUCB (congelado)');
plot(x, Aacc * 100, '-^', 'LineWidth', 2, 'DisplayName', 'LinUCB (adaptativo)');
hold off;
grid on;
xticks(x);
xticklabels(monthLabels);
xlabel('Mes');
ylabel('Accuracy (%)');
title('Desempeño por mes: Softmax vs. LinUCB congelado vs. LinUCB adaptativo');
legend('Location', 'best');

%% GRÁFICA DE MACRO-F1 (menos sensible al dominio de noGesture)
figure('Name', 'Comparación de paradigmas - Macro-F1', 'Position', [100, 650, 800, 500]);
plot(x, Sf1, '-o', 'LineWidth', 2, 'DisplayName', 'Softmax (congelado)');
hold on;
plot(x, Bf1, '-s', 'LineWidth', 2, 'DisplayName', 'LinUCB (congelado)');
plot(x, Af1, '-^', 'LineWidth', 2, 'DisplayName', 'LinUCB (adaptativo)');
hold off;
grid on;
xticks(x);
xticklabels(monthLabels);
xlabel('Mes');
ylabel('Macro-F1');
title('Macro-F1 por mes: Softmax vs. LinUCB congelado vs. LinUCB adaptativo');
legend('Location', 'best');

%% TABLA RESUMEN: ¿HAY DRIFT? ¿ADAPTAR AYUDA?
fprintf('\n========================================================================\n');
fprintf(' RESUMEN: baseline (Mes0-val) vs. %s\n', monthLabels(end));
fprintf('========================================================================\n');

dropSoftmaxAcc = Sacc(1) - Sacc(end);
dropBanditAcc = Bacc(1) - Bacc(end);
dropAdaptiveAcc = Aacc(1) - Aacc(end);

fprintf('Softmax           accuracy: baseline=%.4f  %s=%.4f  caida=%.4f\n', Sacc(1), monthLabels(end), Sacc(end), dropSoftmaxAcc);
fprintf('LinUCB congelado  accuracy: baseline=%.4f  %s=%.4f  caida=%.4f\n', Bacc(1), monthLabels(end), Bacc(end), dropBanditAcc);
fprintf('LinUCB adaptativo accuracy: baseline=%.4f  %s=%.4f  caida=%.4f\n', Aacc(1), monthLabels(end), Aacc(end), dropAdaptiveAcc);

fprintf('\n');
if dropAdaptiveAcc < dropBanditAcc - 0.02
    fprintf('El LinUCB adaptativo cae menos que el congelado (%.4f vs %.4f): la adaptación online con solo la señal de recompensa sí ayuda a compensar el drift.\n', dropAdaptiveAcc, dropBanditAcc);
elseif dropAdaptiveAcc > dropBanditAcc + 0.02
    fprintf('El LinUCB adaptativo cae MÁS que el congelado (%.4f vs %.4f): dejarlo seguir aprendiendo con solo el reward no está ayudando (o el reward shaping necesita revisión).\n', dropAdaptiveAcc, dropBanditAcc);
else
    fprintf('No hay diferencia notable entre congelar el bandit y dejarlo adaptar (caida %.4f vs %.4f).\n', dropBanditAcc, dropAdaptiveAcc);
end
fprintf('========================================================================\n');
