function info = train_one_method(methodObj, Dataset, res_file, trial, ...
    test_iters, opts)

% Training routine for online hashing
%
% INPUTS
%    methodObj - (object)
%      Dataset - (struct) 
%     res_file - (string) path to result file
% 	 trial - (int) trial number
%   test_iters - (int) A vector specifiying the checkpoints, see train.m .
%         opts - (struct) Parameter structure.
%
% OUTPUTS
%  train_time  - (float) elapsed time in learning the hash mapping
%  update_time - (float) elapsed time in updating the hash table
%  reserv_time - (float) elapsed time in maintaing the reservoir set
%  ht_updates  - (int)   total number of hash table updates performed
%  bits_computed - (int) total number of bit recomputations, see update_hash_table.m
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
info.H            = [];
info.W            = [];
info.W_lastupdate = [];
info.update_iters = [];
info.bit_recomp   = 0;
info.time_train   = 0;
info.time_update  = 0;
info.time_reserv  = 0;
%%%%%%%%%%%%%%%%%%%%%%% INIT %%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%% STREAMING BEGINS! %%%%%%%%%%%%%%%%%%%%%%%
num_iters = ceil(opts.numTrain / opts.batchSize);
logInfo('%s: %d train_iters', opts.identifier, num_iters);

for iter = 1:num_iters
    t_ = tic;
    [W, batch] = methodObj.train1batch(W, Xtrain, Ytrain, trainInd, iter, opts);
    info.time_train = info.time_train + toc(t_);

    % ---- reservoir update & compute new reservoir hash table ----
    t_ = tic;
    Hres_new = [];
    if reservoir_size > 0
        % update reservoir
        [reservoir, update_ind] = update_reservoir(reservoir, ...
            Xtrain(batch, :), Ytrain(batch, :), ...
            reservoir_size, W_lastupdate, opts.unsupervised);
        % compute new reservoir hash table (do not update yet)
        Hres_new = (W' * reservoir.X' > 0)';
    end

    % ---- determine whether to update or not ----
    update_table = trigger_update(iter, W_lastupdate, W, reservoir, ...
        Hres_new, opts);
    info.time_reserv = info.time_reserv + toc(t_);

    % ---- hash table update, etc ----
    if update_table
        W_lastupdate = W;  % update snapshot
        info.update_iters = [info.update_iters, iter];
        if reservoir_size > 0
            reservoir.H = Hres_new;
        end

        % recompute hash table
        % NOTE: We only record time for matrix multiplication here as the data 
        %       (Xtrain) is completely loaded in memory. This is not necessarily 
        %       the case with large databases, where disk I/O would probably be 
        %       involved in the hash table recomputation.
        t_ = tic;
        H  = (Xtrain * W_lastupdate)' > 0;
        info.bit_recomp  = info.bit_recomp + prod(size(H));
        info.time_update = info.time_update + toc(t_);
    end

    % ---- CHECKPOINT: save intermediate model ----
    if ismember(iter, test_iters)
        info.W_lastupdate = W_lastupdate;
        info.W = W;
        info.H = H;
        F = sprintf('%s/trial%d_iter%d.mat', opts.expdir, trial, iter);
        save(F, '-struct', 'info');

        logInfo(['*checkpoint*\n[Trial %d] %s\n' ...
            '     (%d/%d) W %.2fs, HT %.2fs (%d updates), Res %.2fs\n' ...
            '     total BR = %g'], ...
            trial, opts.identifier, iter*opts.batchSize, opts.numTrain, ...
            info.time_train, info.time_update, numel(info.update_iters), ...
            info.time_reserv, info.bit_recomp);
    end
end
%%%%%%%%%%%%%%%%%%%%%%% STREAMING ENDED! %%%%%%%%%%%%%%%%%%%%%%%

% save final model
info.ht_updates = numel(info.update_iters);
info.test_iters = test_iters;
info = rmfield(info, 'W_lastupdate');

save(res_file, '-struct', 'info');
logInfo('HT updates: %d, bits computed: %d', info.ht_updates, info.bit_recomp);
logInfo('[Trial %d] Saved: %s\n', trial, res_file);

% rm W/H when returning as stats
info = rmfield(info, 'W');
info = rmfield(info, 'H');

end
