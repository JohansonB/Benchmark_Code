clc; clearvars; close all;
addpath util;
addpath util/distinguishable_colors;
addpath util/subaxis;

X = csvread('./data/mocap_example.csv')';
N = size(X, 2);
D = size(X, 1);

%% PLOT DATA`

col = distinguishable_colors(D);
figure; hold on;
for d=1:D
    plot((1:N)/120, X(d,:), 'Color', col(d,:), 'LineWidth', 2);
end
xlabel('s');
legend({'lhumerus','rhumerus','lfemur','rfemur'});
printpdf(gcf, 'plots/mocap_input.pdf');

%% RUN SUMMARIZATION

% [Smin, Smax, max_dist] = tune_hyperparams(X)

Smin = 150;
Smax = 500;
maxdist = 180;
pred_len = 1000;
verbosity = 0;
signal_freq = 120;

[models, starts, ends, idx, ~, ~] = summarize_seq(X, Smin, Smax, maxdist, verbosity);

%% PLOT SUMMARIZATION
    
num_models = max(idx);
figure('Units', 'pixels', 'Position', [0 0 700 500]);
set(gcf, 'PaperPositionMode', 'auto');
numslots = ceil(1.2*num_models);
sub(1) = subplot(numslots+num_models+2, 1, 2:numslots+1); hold on;  box on;
set(gca,'LineWidth',1);
times = (1:size(X, 2)) / signal_freq;
cols = distinguishable_colors(size(X, 1));
for i=1:length(idx)
    for j=size(X, 1):-1:1
        plot(times(starts(i):ends(i)), X(j, starts(i):ends(i)), 'LineWidth', 2.5, 'Color', cols(j,:)); 
    end
end
xlim([0 30]);
for x1 = starts / signal_freq
  line([x1 x1], get(gca, 'ylim'), 'Color', 'black', 'LineWidth', 1);
end
xlabel('Time (s)');
set(findall(gcf,'Type','Axes'),'FontSize',16);
set(findall(gcf,'Type','Text'),'FontSize',24);
set(findall(gcf,'Type','Legend'),'FontSize',16);

model_cols = distinguishable_colors(num_models);
for i=1:num_models
    fprintf('i=%d\n',i);
    sub(1+i) = subplot(numslots+num_models+2, 1, numslots+i+2); hold on; 
    ylabel(sprintf('%d',i));
    box on; 
    set(gca,'LineWidth',1, 'FontSize',18);
    for j=1:length(idx)
        if idx(j) == i
            st = starts(j)/signal_freq; ed = ends(j)/signal_freq;
            patch([st, ed, ed, st],[0, 0, 1, 1], [.7 .7 .7],'LineWidth',1);
            patch([ed, ed, st, st],[0, 1, 1, 0], [.7 .7 .7],'LineWidth',1);
            set(gca,'XTickLabel',''); set(gca,'YTickLabel','');
        end
    end
end
linkaxes(sub,'x')
printpdf(gcf, 'plots/mocap_summarization.pdf');

%% PLOT MODELS

figure('Units', 'pixels', 'Position', [100 100 150 500]);  hold on;
plot_models_multidim(models);
set(gcf, 'PaperPositionMode', 'auto');
printpdf(gcf,'plots/mocap_models.pdf');