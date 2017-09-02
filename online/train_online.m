function info = train_one_method(methodObj, Dataset, prefix, test_iters, opts)

% Training routine for online hashing
%
% INPUTS
%    methodObj - (object)
%      Dataset - (struct) 
% 	prefix - (string) Prefix of the "checkpoint" files.
%   test_iters - (int)    A vector specifiying the checkpoints, see train.m .
%         opts - (struct) Parameter structure.
%
% OUTPUTS
%  train_time  - (float) elapsed time in learning the hash mapping
%  update_time - (float) elapsed time in updating the hash table
%  res_time    - (float) elapsed time in maintaing the reservoir set
%  ht_updates  - (int)   total number of hash table updates performed
%  bit_computed_all - (int) total number of bit recomputations, see update_hash_table.m
% 

%%%%%%%%%%%%%%%%%%%%%%% INIT %%%%%%%%%%%%%%%%%%%%%%%
Xtrain = Dataset.Xtrain;
Ytrain = Dataset.Ytrain;

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
assert(opts.numTrain<=ntrain, sprintf('opts.numTrain > %d!', ntrain));
trainInd = [];
for e = 1:opts.epoch
    trainInd = [trainInd, randperm(ntrain, opts.numTrain)];
end
opts.numTrain = numel(trainInd);

info = [];
info.bits_computed_all = 0;
info.update_iters = [];
info.train_time  = 0;  
info.update_time = 0;
info.res_time    = 0;
%%%%%%%%%%%%%%%%%%%%%%% INIT %%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%% STREAMING BEGINS! %%%%%%%%%%%%%%%%%%%%%%%
num_iters = ceil(opts.numTrain / opts.batchSize);
logInfo('%s: %d train_iters', opts.identifier, num_iters);

for iter = 1:num_iters
    t_ = tic;
    [W, batchInd] = methodObj.train1batch(W, Xtrain, Ytrain, trainInd, iter, opts);
    train_time = train_time + toc(t_);

    % ---- reservoir update & compute new reservoir hash table ----
    t_ = tic;
    Hres_new = [];
    if reservoir_size > 0
        % update reservoir
        [reservoir, update_ind] = update_reservoir(reservoir, ...
            Xtrain(batchInd, :), Ytrain(batchInd, :), ...
            reservoir_size, W_lastupdate, opts.unsupervised);
        % compute new reservoir hash table (do not update yet)
        Hres_new = (W' * reservoir.X' > 0)';
    end

    % ---- determine whether to update or not ----
    update_table = trigger_update(iter, W_lastupdate, W, reservoir, ...
        Hres_new, opts);
    res_time = res_time + toc(t_);

    % ---- hash table update, etc ----
    if update_table
        W_lastupdate = W;
        update_iters = [update_iters, iter];
        if reservoir_size > 0
            reservoir.H = Hres_new;
        end

        % actual hash table update
        t_ = tic;
        H  = (Xtrain * W_lastupdate)' > 0;
        bits_computed = prod(size(H));
        bits_computed_all = bits_computed_all + bits_computed;
        update_time = update_time + toc(t_);
    end

    % ---- CHECKPOINT: save intermediate model ----
    if ismember(iter, test_iters)
        F = sprintf('%s/%s_iter%d.mat', opts.expdir, prefix, iter);
        save(F, 'W', 'W_lastupdate', 'H', 'bits_computed_all', ...
            'train_time', 'update_time', 'res_time', 'update_iters');

        logInfo(['*checkpoint*\n[%s] %s\n' ...
            '     (%d/%d) W %.2fs, HT %.2fs (%d updates), Res %.2fs\n' ...
            '     total BR = %g'], ...
            prefix, opts.identifier, iter*opts.batchSize, opts.numTrain, ...
            train_time, update_time, numel(update_iters), res_time, ...
            bits_computed_all);
    end
end
%%%%%%%%%%%%%%%%%%%%%%% STREAMING ENDED! %%%%%%%%%%%%%%%%%%%%%%%

% save final model
F = sprintf('%s/%s.mat', opts.expdir, prefix);
save(F, 'W', 'H', 'bits_computed_all', ...
    'train_time', 'update_time', 'res_time', 'test_iters', 'update_iters');

ht_updates = numel(update_iters);
logInfo('%d Hash Table updates, bits computed: %g', ht_updates, bits_computed_all);
logInfo('[%s] Saved: %s\n', prefix, F);
end
