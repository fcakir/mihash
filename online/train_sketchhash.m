function [train_time, update_time, res_time, ht_updates, bits_computed_all] = ...
    train_sketch(Xtrain, Ytrain, thr_dist, prefix, test_iters, trialNo, opts)
% Implementation of the SketchHash method as described in: 
%
% C. Leng, J. Wu, J. Cheng, X. Bai and H. Lu
% "Online Sketching Hashing"
% Computer Vision and Pattern Recognition (CVPR) 2015
%
% Training routine for SketchHash method.
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
%	opts   - (struct)Parameter structure.
%
% OUTPUTS
%  train_time  - (float) elapsed time in learning the hash mapping
%  update_time - (float) elapsed time in updating the hash table
%  res_time    - (float) elapsed time in maintaing the reservoir set
%  ht_updates  - (int)   total number of hash table updates performed
%  bit_computed_all - (int) total number of bit recomputations, see update_hash_table.m
 
% NOTES
% 	W is d x b where d is the dimensionality 
%            and b is the bit length / # hash functions
%

%%%%%%%%%%%%%%%%%%%%%%% GENERIC INIT %%%%%%%%%%%%%%%%%%%%%%%

% set up reservoir
reservoir = [];
reservoir_size = opts.reservoirSize;
if reservoir_size > 0
    reservoir.size = 0;
    reservoir.X    = [];
    if opts.unsupervised
	reservoir.Y = [];
    else
        reservoir.Y = zeros(0, size(Ytrain, 2));
    end
    reservoir.PQ   = [];
    reservoir.H    = [];  % mapped binary codes for the reservoir
end

% order training examples
train_ind = zeros(1, opts.epoch*opts.noTrainingPoints);
for e = 1:opts.epoch
    % randomly shuffle training points before taking first noTrainingPoints
    train_ind((e-1)*opts.noTrainingPoints+1:e*opts.noTrainingPoints) = ...
        randperm(size(Xtrain, 1), opts.noTrainingPoints);
end
opts.noTrainingPoints = opts.noTrainingPoints*opts.epoch;
%%%%%%%%%%%%%%%%%%%%%%% GENERIC INIT %%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%% SET UP SketchHash %%%%%%%%%%%%%%%%%%%%%%%
% convert parameters from opts to internal ones
kInstFeatDimCnt = size(Xtrain, 2);  % feature dim
bits = opts.nbits;
assert(opts.sketchSize <= kInstFeatDimCnt, ...
    sprintf('Somehow, sketching needs sketchSize<=d(%d)', kInstFeatDimCnt));

% initialize hash functions & table
if 0
    % original init for SketchHash, which performed worse
    W = rand(kInstFeatDimCnt, bits) - 0.5;
else
    % LSH init
    d = kInstFeatDimCnt;
    W = randn(d, bits);
    W = W ./ repmat(diag(sqrt(W'*W))',d,1);
end
% NOTE: W_lastupdate keeps track of the last W used to update the hash table
%       W_lastupdate is NOT the W from last iteration
W_lastupdate = W;
stepW = zeros(size(W));  % Gradient accumulation matrix
H = [];  % initial hash table

% for recording time
train_time  = 0;
update_time = 0;
res_time    = 0;

% bits computed
bits_computed_all = 0;

% HT updates
update_iters = [];

% prepare to run online sketching hashing
if opts.noTrainingPoints > 0
    numUseToTrain = opts.noTrainingPoints;
else
    numUseToTrain = size(Xtrain, 1);
end
batchsize      = opts.batchSize;
batchCnt       = ceil(numUseToTrain/batchsize);
instCntSeen    = 0;
instFeatAvePre = zeros(1, kInstFeatDimCnt);  % mean vector
instFeatSkc    = [];
logInfo('%d batches of size %d, sketchSize=%d', batchCnt, batchsize, opts.sketchSize);
%%%%%%%%%%%%%%%%%%%%%%% SET UP SketchHash %%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%% STREAMING BEGINS! %%%%%%%%%%%%%%%%%%%%%%%
for batchInd = 1 : batchCnt

    %%%%%%%%%% LOAD BATCH DATA - BELOW %%%%%%%%%%
    ind = (batchInd-1)*batchsize + 1 : min(batchInd*batchsize, numUseToTrain);
    instFeatInBatch = Xtrain(ind, :);

    instCntInBatch = size(instFeatInBatch, 1);
    %%%%%%%%%% LOAD BATCH DATA - ABOVE %%%%%%%%%%


    %%%%%%%%%% UPDATE HASHING FUNCTION - BELOW %%%%%%%%%%
    tic;

    % calculate current mean feature vector
    instFeatAveCur = mean(instFeatInBatch, 1);

    % sketech current training batch
    if batchInd == 1
        instFeatToSkc = bsxfun(@minus, instFeatInBatch, instFeatAveCur);
    else
        instFeatCmps = sqrt(instCntSeen * instCntInBatch / (instCntSeen + instCntInBatch)) * (instFeatAveCur - instFeatAvePre);
        instFeatToSkc = [bsxfun(@minus, instFeatInBatch, instFeatAveCur); instFeatCmps];
    end
    instFeatSkc = MatrixSketch_Incr(instFeatSkc, instFeatToSkc, opts.sketchSize);

    % update mean feature vector and instance counter
    instFeatAvePre = (instFeatAvePre * instCntSeen + instFeatAveCur * instCntInBatch) / (instCntSeen + instCntInBatch);
    instCntSeen = instCntSeen + instCntInBatch;

    % compute QR decomposition of the sketched matrix
    [q, r] = qr(instFeatSkc', 0);
    [u, ~, ~] = svd(r, 'econ');
    v = q * u;

    % obtain the original projection matrix
    hashProjMatOrg = v(:, 1 : bits);

    % use random rotation
    R = orth(randn(bits));

    % update hashing function
    W = hashProjMatOrg * R;

    train_time = train_time + toc;
    %%%%%%%%%% UPDATE HASHING FUNCTION - ABOVE %%%%%%%%%%


    % ---- reservoir update & compute new reservoir hash table ----
    t_ = tic;
    Hres_new = [];
    if reservoir_size > 0
        Xs = bsxfun(@minus, instFeatInBatch, instFeatAvePre);
	if ~isempty(Ytrain), Ys = Ytrain(ind, :); else, Ys = []; end;
        [reservoir, update_ind] = update_reservoir(reservoir, Xs, Ys, ...
            reservoir_size, W_lastupdate, opts.unsupervised);
        % compute new reservoir hash table (do not update yet)
        Hres_new = (W' * reservoir.X' > 0)';
    end


    % ---- determine whether to update or not ----
    if batchInd*opts.batchSize <= opts.sketchSize
        % special for SketchHash: fill sketch matrix first
        update_table = true;
        trigger_val  = 0;
    else
        [update_table, trigger_val] = trigger_update(batchInd, opts, ... 
            W_lastupdate, W, reservoir, Hres_new, opts.unsupervised, thr_dist);
    end
    res_time = res_time + toc(t_);


    % ---- hash table update, etc ----
    if update_table
        W_lastupdate = W;
        update_iters = [update_iters, batchInd];

        % update reservoir hash table
        if reservoir_size > 0
            reservoir.H = Hres_new;
        end

        % actual hash table update (record time)
        t_ = tic;
        X_cent = bsxfun(@minus, Xtrain, instFeatAvePre);  % centering
        [H, bits_computed] = update_hash_table(H, W_lastupdate, ...
            X_cent, Ytrain, update_iters, opts);
        bits_computed_all = bits_computed_all + bits_computed;
        update_time = update_time + toc(t_);
    end


    % ---- cache intermediate model to disk ----
    % CHECKPOINT
    if ismember(batchInd, test_iters)
        F = sprintf('%s_iter%d.mat', prefix, batchInd);
        save(F, 'W', 'W_lastupdate', 'H', 'bits_computed_all', ...
            'train_time', 'update_time', 'res_time', 'update_iters');

        logInfo(['*checkpoint*\n[T%02d] %s\n' ...
            '     (%d/%d) W %.2fs, HT %.2fs (%d updates), Res %.2fs\n' ...
            '     total BR = %g'], ...
            trialNo, opts.identifier, batchInd*opts.batchSize, opts.noTrainingPoints, ...
            train_time, update_time, numel(update_iters), res_time, ...
            bits_computed_all);
    end
end
%%%%%%%%%%%%%%%%%%%%%%% STREAMING ENDED! %%%%%%%%%%%%%%%%%%%%%%%

% save final model, etc
F = [prefix '.mat'];
save(F, 'instFeatAvePre', 'W', 'H', 'bits_computed_all', ...
    'train_time', 'update_time', 'res_time', 'test_iters', 'update_iters');

ht_updates = numel(update_iters);
logInfo('%d Hash Table updates, bits computed: %g', ht_updates, bits_computed_all);
logInfo('[T%02d] Saved: %s\n', trialNo, F);
end
