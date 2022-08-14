% This function forecasts data based on a summarization. Note that it runs summarize_seq first, 
% so it returns a lot of the same output. 
% 
% Input: 
% X (a d x N time series), where d is the dimensionality and N is length
% pred_len (number of time ticks to predict)
% Smin (min segment length)
% Smax (max segment length)
% max_dist (max warping distance, i.e. Sakoe Chiba band width)
% verbose (verbosity level: 0 for low and 1 for high)
% 
% Output: 
% Xp (forecasted time series of size d x pred_len)
% idx (index of vocabulary term assigned to each segment)
% starts (starts of each segment)
% ends (ends of each segment)
% p_idx (same as idx, but corresponding to the forecasted region)
% p_starts (same as starts, but corresponding to the forecasted region)
% p_ends (same as ends, but corresponding to the forecasted region)
% models (a list of vocabulary terms)
% 
function [Xp,time, idx, starts, ends, p_idx, p_starts, p_ends, models] = forecast_seq(X, pred_len, Smin, Smax, max_dist, verbosity)

tic;
ndim = size(X, 1); 
N = size(X, 2);

tot_len = pred_len + N;
[models, starts, ends, idx, best_prefix_length, ~] = summarize_seq(X, Smin, Smax, max_dist, verbosity);
fprintf('prefix length: %d\n', best_prefix_length);

%%
Xp = X; 
p_starts = [];
p_ends = [];
suffix = models{idx(end)}(:, best_prefix_length+1:end);
if ~isempty(suffix)
    Xp = [Xp bsxfun(@plus, suffix, X(:, end)-suffix(:, 1))];
    p_starts(1) = size(X,2) + 1; 
    p_ends(1) = size(Xp, 2);
end
m = init_markov(3);
for i=1:length(idx)
    m = update_markov(m, idx(1:i));
end
p_idx = [];
while size(Xp, 2) < tot_len
    best_char = predict_markov(m, [idx p_idx]);
    p_idx = [p_idx best_char];
    p_starts(length(p_starts)+1) = length(Xp)+1;
%     Xp = [Xp bsxfun(@plus, models{best_char}, mean(X, 2))];
    Xp = [Xp models{best_char}];
    p_ends(length(p_ends)+1) = length(Xp);
end
if ~isempty(suffix)
    p_idx = [idx(end) p_idx];
end
Xp = Xp(:, length(X)+1:tot_len);
p_ends(end) = tot_len;
time = toc;
