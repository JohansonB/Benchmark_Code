% summarize_seq: This function summarizes the given time series (i.e. learning 
% a segmentation, vocabulary terms, and an assignment of which vocabulary term 
% should be used to describe each segment).
%
% Input: 
% X (a d by N time series), where d is the dimensionality and N is length
% Smin (min segment length)
% Smax (max segment length)
% max_dist (max warping distance, i.e. Sakoe Chiba band width)
% verbose (verbosity level: 0 for low and 1 for high)
% 
% Output: 
% models (a list of vocabulary terms)
% starts (starts of each segment)
% ends (ends of each segment)
% idx (index of vocabulary term assigned to each segment)
% best_prefix_length (can usually be ignored; this indicates the partial matching of a vocabulary term at the end of the data)
% tot_err (total error in terms of description length)

function [models, starts, ends, idx, best_prefix_length, tot_err] = summarize_seq(X, Smin, Smax, max_dist, verbose)

% X = std(X(1,:)) * bsxfun(@rdivide, X, std(X, [], 2));

% Xm = bsxfun(@minus, X, mean(X, 2));
% [b,a] = butter(1, [1]/(180), 'high');
% X = filter(b,a,Xm);

starts = [];
ends = [];
starts(1) = 1;
[best_initial, ~] = new_segment_size(X, 1, {}, Smin, Smax, max_dist);
ends(1) = best_initial;
models = {X(:, starts(1) : ends(1))};
idx = [1];
model_momentum = .8;
max_vocab = 5;
termination_threshold = 0; % if we have less than this length left, stop

new_cluster_threshold = .3; 
% fraction of unexplained variance needed to make new cluster (lower = more clusters)
% mean_abs_dev = mean(abs(Xs - mean(Xs)));
mean_dev = mean((X(:) - mean(X(:))).^2);

best_prefix_length = nan;
tot_err = 0;
while ends(end) + termination_threshold < size(X, 2)
    cur_idx = length(starts) + 1;
    cur = ends(end) + 1;
    starts(cur_idx) = cur;
    %fprintf('SEGMENT %d at position %d ======== \n', cur_idx, cur);
    
    num_models = length(models);
    ave_costs = inf(num_models, Smax);
    
    cur_end = min(cur + Smax - 1, size(X, 2));
    Xcur = X(:, cur : cur_end);
    for k = 1:num_models
        [dtw_dist, dtw_mat, ~, dtw_trace] = dtw(models{k}, Xcur, max_dist);
        dtw_costs = dtw_mat(end, :);
        
        ave_costs(k, 1:size(Xcur, 2)) = dtw_costs ./ (1 : size(Xcur, 2));
        ave_costs(k, 1:Smin-1) = nan;
    end
    
    [best_cost, best_idx] = min(ave_costs(:));
    [best_k, best_size] = ind2sub(size(ave_costs), best_idx);
   
    
    if cur + Smax >= size(X, 2) % match prefixes of models to rest of X
        good_prefix_costs = nan(num_models, 1); % best prefix cost for each model
        good_prefix_lengths = nan(num_models, 1);
        for k = 1:num_models
            [~, dtw_mat, ~, ~] = dtw(models{k}, Xcur, max_dist);
            prefix_costs = dtw_mat(:, end)'; % try all prefixes of models{k}
            [model_dim,model_length] = size(models{k});
            ave_prefix_costs = prefix_costs ./ (1:model_length);
            [good_prefix_costs(k), good_prefix_lengths(k)] = min(ave_prefix_costs);
       
        end
        [best_prefix_cost, best_prefix_k] = min(good_prefix_costs);
        best_prefix_length = good_prefix_lengths(best_prefix_k);
        
      
        %fprintf('end state: best k is %d, best cost is %.3f, best prefix cost is %.3f\n', best_k, best_cost, best_prefix_cost);
        if best_prefix_cost < best_cost
            %fprintf('ending with prefix\n');
            ends(cur_idx) = length(X);
            idx(cur_idx) = best_prefix_k;
            break;
        end
    end
    
    %fprintf('cluster costs: %.2f\n', ave_costs(:, best_size));
%     c1_costs = [c1_costs ave_costs(1, best_size)];
    %fprintf('new cluster costs for %d: %.2f\n', size(X, 1), new_cluster_threshold * mean_dev * size(X, 1));
    %fprintf('size chosen = %d\n', best_size);
    Xbest = X(:, cur : cur + best_size - 1);
    if best_cost > new_cluster_threshold * mean_dev && length(models) < max_vocab
        %fprintf('=> new cluster\n');
        [best_S1, ~] = new_segment_size(X, cur, models, Smin, Smax, max_dist);
        ends(cur_idx) = cur + best_S1 - 1;
        idx(cur_idx) = num_models + 1;
        models{num_models + 1} = X(:, starts(cur_idx) : ends(cur_idx));
        tot_err = tot_err + new_cluster_threshold * mean_dev * best_S1;
    else
        %fprintf('=> cluster %d\n', best_k);
        ends(cur_idx) = cur + best_size - 1;
        idx(cur_idx) = best_k;
        tot_err = tot_err + best_cost * best_size;
        
        [~, ~, ~, dtw_trace] = dtw(models{best_k}, Xbest, max_dist);
        trace_summed = zeros(size(models{best_k}));
        for t = 1:size(dtw_trace, 1)
            trace_summed(:, dtw_trace(t, 1)) = trace_summed(:, dtw_trace(t, 1)) + Xbest(:, dtw_trace(t, 2));
        end
        trace_counts = tabulate(dtw_trace(:, 1));
        trace_counts = trace_counts(:, 2)';
        trace_ave = bsxfun(@rdivide, trace_summed, trace_counts);
        models{best_k} = model_momentum * models{best_k} + (1 - model_momentum) * trace_ave;
    end
    
    if verbose >= 1
        [~, ~, ~, dtw_trace] = dtw(models{best_k}, Xbest, max_dist);
        figure('Units', 'pixels', 'Position', [100 100 400 500]); hold on;
        linestyle = {'-','--'};
        for j=1:size(models{best_k},1)
            plot(models{best_k}(j,:), 'x-', 'Marker', 'o', 'LineWidth', 2.5, 'Color','red', 'LineStyle', linestyle{j});
            plot(Xbest(j,:), '-', 'LineWidth', 2, 'Color','black', 'LineStyle', linestyle{j});
            plot(dtw_trace(:, 1), Xbest(j, dtw_trace(:, 2)), 'x-', 'LineWidth', 1.5, 'Color','blue', 'LineStyle', linestyle{j});
        end
        title(sprintf('SEGMENT %d ======== cost=%.3f\n', cur_idx,best_cost));
        hold off;
        figure; plot_models(models, idx); hold off;
    end
end
tot_err = tot_err/std(X(:))^2 + (length(idx)-1) * log2(length(X)) + length(idx) * log2(length(models));
% c1_costs = c1_costs(1:10);
% c1_costs
% fprintf(sprintf('ave: %.3f\n', mean(c1_costs)));
% end