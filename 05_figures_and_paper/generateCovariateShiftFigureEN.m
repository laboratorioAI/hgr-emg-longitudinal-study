% #################################################################
% English-language version of the MMD^2 / A-distance covariate-shift
% figure, for the ICASSP 2027 paper (RL_vs_SL/ICASSP2027-paper.tex).
% Reuses the already-computed statistics from evaluateCovariateShift.m
% (Models/CovariateShift_29-07-2026_12-28-56.mat) -- no recomputation,
% only re-labeling for an English-reading audience.
% #################################################################

clear; clc;
s = load('Models/CovariateShift_29-07-2026_12-28-56.mat');
r = s.results;
numMeses = numel(r);

mmdVals = [r.mmd2];
pMMD = [r.pValuePermutacion];
aDistVals = [r.aDistance];
pDomain = [r.pValueDomainClassifier];
monthLabels = arrayfun(@(m) sprintf('Month%d', m), 1:numMeses, 'UniformOutput', false);

fig = figure('Visible', 'off', 'Position', [100 100 700 1000]);

subplot(2,1,1);
bar(1:numMeses, mmdVals, 'FaceColor', [0.34 0.61 0.84]);
hold on;
for m = 1:numMeses
    if pMMD(m) < 0.05
        text(m, mmdVals(m) + 0.05*max(mmdVals), '*', 'HorizontalAlignment', 'center', 'FontSize', 22, 'FontWeight', 'bold');
    end
end
hold off;
xticks(1:numMeses); xticklabels(monthLabels);
ylabel('MMD^2 (Month0 vs MonthN)');
title('MMD^2 (* = p<0.05, permutation test)');
grid on;
set(gca, 'FontSize', 16);
set(get(gca,'XLabel'), 'FontSize', 17);
set(get(gca,'YLabel'), 'FontSize', 17);
set(get(gca,'Title'), 'FontSize', 18);

subplot(2,1,2);
bar(1:numMeses, aDistVals, 'FaceColor', [0.93 0.49 0.19]);
hold on;
for m = 1:numMeses
    if pDomain(m) < 0.05
        text(m, aDistVals(m) + 0.05*max(aDistVals), '*', 'HorizontalAlignment', 'center', 'FontSize', 22, 'FontWeight', 'bold');
    end
end
hold off;
xticks(1:numMeses); xticklabels(monthLabels);
ylabel('A-distance (0=identical, 2=fully separable)');
title('A-distance / domain classifier (* = p<0.05)');
grid on;
set(gca, 'FontSize', 16);
set(get(gca,'XLabel'), 'FontSize', 17);
set(get(gca,'YLabel'), 'FontSize', 17);
set(get(gca,'Title'), 'FontSize', 18);

sgtitle('Covariate shift in the representation space (128-d)', 'FontSize', 19, 'FontWeight', 'bold');

outDir = 'RL_vs_SL';
exportgraphics(fig, fullfile(outDir, 'fig_covariate_shift_en.png'), 'Resolution', 200);
close(fig);
fprintf('Saved: %s\n', fullfile(outDir, 'fig_covariate_shift_en.png'));
