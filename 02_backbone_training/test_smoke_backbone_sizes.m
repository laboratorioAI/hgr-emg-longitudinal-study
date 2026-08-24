% Smoke test: construye el layerGraph de las 3 variantes de tamaño
% (grande/mediano/pequeno) usando el código real de producción
% (trainBackboneTransformer.m en modo soloVerificar=true), sin datastores
% ni entrenamiento. Falla con error si alguna topología es inválida.
addpath('CNN-Transformer');
clc;

tamanos = {'grande', 'mediano', 'pequeno'};
for i = 1:numel(tamanos)
    fprintf('\n--- Verificando tamaño: %s ---\n', tamanos{i});
    trainBackboneTransformer(tamanos{i}, true);
    close all;
end
fprintf('\nOK: las 3 variantes de tamaño construyen un grafo valido.\n');
