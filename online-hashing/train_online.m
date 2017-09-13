function info = train_online(methodObj, Dataset, trial, test_iters, opts)

% Training routine for online hashing
%
% INPUTS
%    methodObj - (object)
%      Dataset - (struct) 
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

% hash mapping & hash table
[W, reservoir, Xtrain, methodObj] = methodObj.init(reservoir, Xtrain, Ytrain, opts);
H = methodObj.encode(W, Xtrain, false);
W_snapshot = W;  % snapshot hash mapping

% order training examples
ntrain = size(Xtrain, 1);
assert(opts.numTrain<=ntrain, sprintf('opts.numTrain > %d!', ntrain));
trainInd = [];
for e = 1:opts.epoch
    trainInd = [trainInd, randperm(ntrain, opts.numTrain)];
end
numTrainTotal = numel(trainInd);

% set up reservoir
reservoir = [];
if opts.reservoirSize > 0
    reservoir.size = 0;
    reservoir.PQ   = [];  % priority queue
    reservoir.H    = [];  % hash table for the reservoir
    reservoir.X    = zeros(0, size(Xtrain, 2));
    reservoir.Y    = zeros(0, size(Ytrain, 2));
end

info = [];
info.H            = [];
info.W            = [];
info.W_snapshot   = [];
info.update_iters = [];
info.bit_recomp   = 0;
info.time_train   = 0;
info.time_update  = 0;
info.time_reserv  = 0;

iterdir = fullfile(opts.dirs.exp, sprintf('trial%d_iter', trial));
if ~exist(iterdir), mkdir(iterdir); end
%%%%%%%%%%%%%%%%%%%%%%% INIT %%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%% STREAMING BEGINS! %%%%%%%%%%%%%%%%%%%%%%%
opts.num_iters = ceil(numTrainTotal / opts.batchSize);
logInfo('%s: %d train_iters', opts.identifier, opts.num_iters);

update_table = false;
for iter = 1:opts.num_iters
    t_ = tic;
    [W, batch] = methodObj.train1batch(W, reservoir, Xtrain, Ytrain, ...
        trainInd, iter, opts);
    info.time_train = info.time_train + toc(t_);

    % ---- reservoir update & compute new reservoir hash table ----
    t_ = tic;
    Hres_new = [];
    if opts.reservoirSize > 0
        % update reservoir
        [reservoir, update_ind] = update_reservoir(reservoir, ...
            Xtrain(batch, :), Ytrain(batch, :), ...
            opts.reservoirSize, opts.unsupervised);
        % update reservoir hash table (with snapshot)
        Hres = methodObj.encode(W_snapshot, reservoir.X, true);
        reservoir.H(update_ind, :) = Hres(update_ind, :);
    end

    % ---- determine whether to update or not ----
    if ~mod(iter*opts.batchSize, opts.updateInterval)
        % new reservoir hash table (with new W)
        Hres_new = methodObj.encode(W, reservoir.X, true);
        update_table = trigger_update(iter, W_snapshot, W, reservoir, ...
            Hres_new, opts);
    end
    info.time_reserv = info.time_reserv + toc(t_);

    % ---- hash table update, etc ----
    if update_table
        W_snapshot = W;  % update snapshot
        info.update_iters = [info.update_iters, iter];
        if opts.reservoirSize > 0
            reservoir.H = Hres_new;
        end

        % recompute hash table
        % NOTE: We only record time for matrix multiplication here as the data 
        %       (Xtrain) is completely loaded in memory. This is not necessarily 
        %       the case with large databases, where disk I/O would probably be 
        %       involved in the hash table recomputation.
        t_ = tic;
        H  = methodObj.encode(W_snapshot, Xtrain, false);
        info.bit_recomp  = info.bit_recomp + prod(size(H));
        info.time_update = info.time_update + toc(t_);
        update_table = false;
    end

    % ---- CHECKPOINT: save intermediate model ----
    if ismember(iter, test_iters)
        info.W_snapshot = W_snapshot;
        info.W = W;
        info.H = H;
        F = sprintf('%s/%d.mat', iterdir, iter);
        save(F, '-struct', 'info');

        logInfo('');
        logInfo('[T%d] CHECKPOINT @ iter %d/%d (batchSize %d)', trial, iter, ...
            opts.num_iters, opts.batchSize);
        logInfo('[%s] %s', opts.methodID, opts.identifier);
        logInfo('W %.2fs, HT %.2fs (%d updates), Res %.2fs. #BR: %.3g', ...
            info.time_train, info.time_update, numel(info.update_iters), ...
            info.time_reserv, info.bit_recomp);
        logInfo('');
    end
end
%%%%%%%%%%%%%%%%%%%%%%% STREAMING ENDED! %%%%%%%%%%%%%%%%%%%%%%%

% finalize info
info.ht_updates = numel(info.update_iters);
info.test_iters = test_iters;

info = rmfield(info, 'W_snapshot');
info = rmfield(info, 'W');
info = rmfield(info, 'H');

logInfo('HT updates: %d, bits computed: %d', info.ht_updates, info.bit_recomp);

end
