function info = train_one_method(methodObj, Xtrain, Ytrain, thr_dist, prefix, ...
    test_iters, trialNo, opts)

% Training routine for AdaptHash method, see demo_adapthash.m .
%
% INPUTS
% 	Xtrain - (float) n x d matrix where n is number of points 
%       	         and d is the dimensionality 
%
% 	Ytrain - (int)   n x l matrix containing labels, for unsupervised datasets
% 			 might be empty, e.g., LabelMe.
%     thr_dist - (int)   For unlabelled datasets, corresponds to the distance 
%		         value to be used in determining whether two data instance
% 		         are neighbors. If their distance is smaller, then they are
% 		         considered neighbors.
%	       	         Given the standard setup, this threshold value
%		         is hard-wired to be compute from the 5th percentile 
% 		         distance value obtain through 2,000 training instance.
% 			 see load_gist.m . 
% 	prefix - (string) Prefix of the "checkpoint" files.
%   test_iters - (int)   A vector specifiying the checkpoints, see train.m .
%   trialNo    - (int)   Trial ID
%	opts   - (struct) Parameter structure.
%
% OUTPUTS
%  train_time  - (float) elapsed time in learning the hash mapping
%  update_time - (float) elapsed time in updating the hash table
%  res_time    - (float) elapsed time in maintaing the reservoir set
%  ht_updates  - (int)   total number of hash table updates performed
%  bit_computed_all - (int) total number of bit recomputations, see update_hash_table.m
% 
% NOTES
% 	W is d x b where d is the dimensionality 
%            and b is the bit length / # hash functions
%
% 	If number_iterations is 1000, this means 2000 points will be processed, 
% 	data arrives in pairs

%%%%%%%%%%%%%%%%%%%%%%% INIT %%%%%%%%%%%%%%%%%%%%%%%

H = [];  % hash table (mapped binary codes)
W = methodObj.init(Xtrain, opts);  % hash mapping

% keep track of the last W used to update the hash table
% NOTE: W_lastupdate is NOT the W from last iteration
W_lastupdate = W;  

% set up reservoir
reservoir = [];
reservoir_size = opts.reservoirSize;
if reservoir_size > 0
    reservoir.size = 0;
    reservoir.PQ   = [];
    reservoir.H    = [];  % hash table for the reservoir
    reservoir.X    = zeros(0, size(Xtrain, 2));
    if opts.unsupervised
	reservoir.Y = [];
    else
        reservoir.Y = zeros(0, size(Ytrain, 2));
    end
end

% order training examples
ntrain = size(Xtrain, 1);
train_ind = [];
for e = 1:opts.epoch
    % randomly shuffle training points before taking first noTrainingPoints
    train_ind = [train_ind, randperm(ntrain, opts.noTrainingPoints)];
end
opts.noTrainingPoints = numel(train_ind);

info = [];
info.bits_computed_all = 0;
info.update_iters = [];
info.train_time  = 0;  
info.update_time = 0;
info.res_time    = 0;
%%%%%%%%%%%%%%%%%%%%%%% INIT %%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%% STREAMING BEGINS! %%%%%%%%%%%%%%%%%%%%%%%
for iter = 1:number_iterations
    t_ = tic;
    % TODO check input/output
    [W, sampleIdx] = methodObj.train1batch(W, Xtrain, Ytrain, train_ind, iter, opts);
    train_time = train_time + toc(t_);

    % ---- reservoir update & compute new reservoir hash table ----
    t_ = tic;
    Hres_new = [];
    if reservoir_size > 0
        Xsample = Xtrain(sampleIdx, :);
        Ysample = Ytrain(sampleIdx, :);
        [reservoir, update_ind] = update_reservoir(reservoir, Xsample, Ysample, ...
            reservoir_size, W_lastupdate, opts.unsupervised);
        % compute new reservoir hash table (do not update yet)
        Hres_new = (W' * reservoir.X' > 0)';
    end

    % ---- determine whether to update or not ----
    [update_table, trigger_val] = trigger_update(iter, ...
        opts, W_lastupdate, W, reservoir, Hres_new, ...
        opts.unsupervised, thr_dist);
    res_time = res_time + toc(t_);

    % ---- hash table update, etc ----
    if update_table
        W_lastupdate = W;
        update_iters = [update_iters, iter];

        % update reservoir hash table
        if reservoir_size > 0
            reservoir.H = Hres_new;
        end

        % actual hash table update (record time)
        t_ = tic;
        [H, bits_computed] = update_hash_table(H, W_lastupdate, ...
            Xtrain, Ytrain, update_iters, opts);
        bits_computed_all = bits_computed_all + bits_computed;
        update_time = update_time + toc(t_);
    end

    % ---- save intermediate model ----
    % CHECKPOINT
    if ismember(iter, test_iters)
        F = sprintf('%s_iter%d.mat', prefix, iter);
        save(F, 'W', 'W_lastupdate', 'H', 'bits_computed_all', ...
            'train_time', 'update_time', 'res_time', 'update_iters');

        logInfo(['*checkpoint*\n[T%02d] %s\n' ...
            '     (%d/%d) W %.2fs, HT %.2fs (%d updates), Res %.2fs\n' ...
            '     total BR = %g'], ...
            trialNo, opts.identifier, iter*opts.batchSize, opts.noTrainingPoints, ...
            train_time, update_time, numel(update_iters), res_time, ...
            bits_computed_all);
    end
end
%%%%%%%%%%%%%%%%%%%%%%% STREAMING ENDED! %%%%%%%%%%%%%%%%%%%%%%%

% save final model, etc
F = [prefix '.mat'];
save(F, 'W', 'H', 'bits_computed_all', ...
    'train_time', 'update_time', 'res_time', 'test_iters', 'update_iters');

ht_updates = numel(update_iters);
logInfo('%d Hash Table updates, bits computed: %g', ht_updates, bits_computed_all);
logInfo('[T%02d] Saved: %s\n', trialNo, F);
end
