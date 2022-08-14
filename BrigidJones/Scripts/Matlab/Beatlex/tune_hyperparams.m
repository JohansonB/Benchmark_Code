% Automatically tune the min and max period size (Smin and Smax)

function [best_Smin, best_Smax, best_maxdist] = tune_hyperparams(X)

num_Smax_cand = 6;
trial_params = [];
Smax_cand = round(logspace(2, log10(length(X)/10), num_Smax_cand));
scores = [];

for Smax = Smax_cand
    Smin_cand = round([Smax/4 Smax/3 Smax/2]);
    for Smin = Smin_cand
        maxdist_cand = round([(Smax - Smin)/2 (Smax - Smin)]);
        for maxdist = maxdist_cand
            trial_params = [trial_params; [Smin Smax maxdist]];
            [~, ~, ~, ~, ~, tot_err] = summarize_seq(X, Smin, Smax, maxdist, 0);
            scores = [scores tot_err];
        end
    end
end

[~, best_idx] = min(scores(:));
best_param = trial_params(best_idx, :);
best_Smin = best_param(1);
best_Smax = best_param(2);
best_maxdist = best_param(3);