function [train_time, update_time, res_time, ht_updates, bits_computed_all, bitflips] = ...
    train_sketch(Xtrain, Ytrain, thr_dist, prefix, test_iters, trialNo, opts)

%%%%%%%%%%%%%%%%%%%%%%% GENERIC INIT %%%%%%%%%%%%%%%%%%%%%%%
% are we handling a mult-labeled dataset?
multi_labeled = (size(Ytrain, 2) > 1);
if multi_labeled, myLogInfo('Handling multi-labeled dataset'); end

% set up reservoir
reservoir = [];
reservoir_size = opts.reservoirSize;
if reservoir_size > 0
    reservoir.size = 0;
    reservoir.X    = [];
    if opts.unsupervised
	reservoir.Y = [];
    else
        reservoir.Y  = zeros(0, size(Ytrain, 2));
    end
    reservoir.PQ   = [];
    reservoir.H    = [];  % mapped binary codes for the reservoir
end

% order training examples
if opts.pObserve > 0
    % [OPTIONAL] order training points according to label arrival strategy
    train_ind = get_ordering(trialNo, Ytrain, opts);
else
    % randomly shuffle training points before taking first noTrainingPoints
    train_ind = randperm(size(Xtrain, 1), opts.noTrainingPoints);
end
%%%%%%%%%%%%%%%%%%%%%%% GENERIC INIT %%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%% SET UP SketchHash %%%%%%%%%%%%%%%%%%%%%%%
% convert parameters from opts to internal ones
kInstFeatDimCnt = size(Xtrain, 2);  % feature dim
bits = opts.nbits;

% initialize hash functions & table
if 0
    % original init for SketchHash
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
train_time = 0;
update_time = 0;
res_time = 0;

% bit flips & bits computed
bitflips          = 0;
bitflips_res      = 0;
bits_computed_all = 0;

% HT updates
update_iters = [];
h_ind_array  = [];

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
myLogInfo('%d batches of size %d', batchCnt, batchsize);
%%%%%%%%%%%%%%%%%%%%%%% SET UP SketchHash %%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%% STREAMING BEGINS! %%%%%%%%%%%%%%%%%%%%%%%
for batchInd = 1 : batchCnt

    %%%%%%%%%% LOAD BATCH DATA - BELOW %%%%%%%%%%

    ind = (batchInd-1)*batchsize + 1 : min(batchInd*batchsize, numUseToTrain);
    instFeatInBatch = Xtrain(ind, :);

    instCntInBatch = size(instFeatInBatch, 1);

    %timeElpsStr.loadBtchData(batchInd) = toc;
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

    %for ind = 1 : numel(kLoopBitsLst)
    % obtain the length of hashing code
    %bits = kLoopBitsLst(ind);

    % obtain the original projection matrix
    hashProjMatOrg = v(:, 1 : bits);

    % use random rotation
    R = orth(randn(bits));

    % update hashing function
    %hashProjMat{ind} = single(hashProjMatOrg * R);
    W = hashProjMatOrg * R;
    %end

    %timeElpsStr.updtHashFunc(batchInd) = toc;
    train_time = train_time + toc;

    %%%%%%%%%% UPDATE HASHING FUNCTION - ABOVE %%%%%%%%%%



    %%%%%%%%%% COMPUTE HASHING CODE - BELOW %%%%%%%%%%
    %tic;

    % compute centered query/database instances
    %instFeatQryCen = bsxfun(@minus, instFeatQry, instFeatAvePre);
    %instFeatDtbCen = bsxfun(@minus, instFeatDtb, instFeatAvePre);
    %instFeatDtbCen = bsxfun(@minus, Xtrain, instFeatAvePre);

    %{
    % compute hash code for query/database subset
    for ind = 1 : numel(kLoopBitsLst)
        bits = kLoopBitsLst(ind);

        %codeQry = (instFeatQryCen * hashProjMat{ind} > 0);
        %codeDtb = (instFeatDtbCen * hashProjMat{ind} > 0);

        % save hash code for query/database subset
        codeQryPath = sprintf('%s/codeQry.%d.mat', codeDirCur, bits);
        codeDtbPath = sprintf('%s/codeDtb.%d.mat', codeDirCur, bits);
        save(codeQryPath, 'codeQry');
        save(codeDtbPath, 'codeDtb');
    end
    timeElpsStr.calcHashCode(batchInd) = toc;
    %}
    %%%%%%%%%% COMPUTE HASHING CODE - ABOVE %%%%%%%%%%


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
    [update_table, trigger_val, h_ind] = trigger_update(batchInd, ...
        opts, W_lastupdate, W, reservoir, Hres_new);
    inv_h_ind = setdiff(1:opts.nbits, h_ind);  % keep these bits unchanged
    if reservoir_size > 0 && numel(h_ind) < opts.nbits  % selective update
        assert(opts.fracHash < 1);
        Hres_new(:, inv_h_ind) = reservoir.H(:, inv_h_ind);
    end
    res_time = res_time + toc(t_);


    % ---- hash table update, etc ----
    if update_table
        h_ind_array = [h_ind_array; single(ismember(1:opts.nbits, h_ind))];
        W_lastupdate(:, h_ind) = W(:, h_ind);
        if opts.accuHash <= 0
            W = W_lastupdate;
            myLogInfo('not accumulating gradients!');
        end
        if opts.fracHash < 1
            myLogInfo('selective update: fracHash=%g, randomHash=%g', ...
                opts.fracHash, opts.randomHash);
        end
        update_iters = [update_iters, batchInd];

        % update reservoir hash table
        if reservoir_size > 0
            reservoir.H = Hres_new;
            if strcmpi(opts.trigger, 'bf')
                bitflips_res = bitflips_res + trigger_val;
            end
        end

        % actual hash table update (record time)
        t_ = tic;
        X_cent = bsxfun(@minus, Xtrain, instFeatAvePre);  % centering
        [H, bf_all, bits_computed] = update_hash_table(H, W_lastupdate, ...
            X_cent, Ytrain, h_ind, update_iters, opts);
        bits_computed_all = bits_computed_all + bits_computed;
        bitflips = bitflips + bf_all;
        update_time = update_time + toc(t_);
    end


    % ---- cache intermediate model to disk ----
    %if (~opts.onlyfinal && ismember(batchInd, test_batchInds)) || ...
            %(opts.onlyfinal && batchInd==batchCnt)
    if ismember(batchInd, test_iters)
        F = sprintf('%s_iter%d.mat', prefix, batchInd);
        save(F, 'W', 'W_lastupdate', 'H', 'bitflips', 'bits_computed_all', ...
            'train_time', 'update_time', 'res_time', 'update_iters');
        if ~opts.windows, unix(['chmod o-w ' F]); end  % matlab permission bug
        myLogInfo('[T%02d] batch%d/%d Func %.2fs, Table %.2fs #BF=%g', ...
            trialNo, batchInd, batchCnt, train_time, update_time, bitflips);
    end
end
%%%%%%%%%%%%%%%%%%%%%%% STREAMING ENDED! %%%%%%%%%%%%%%%%%%%%%%%

% save final model, etc
F = [prefix '.mat'];
save(F, 'instFeatAvePre', 'W', 'H', 'bitflips', 'bits_computed_all', ...
    'train_time', 'update_time', 'res_time', 'test_iters', 'update_iters', ...
    'h_ind_array');
% fix permission
if ~opts.windows, unix(['chmod g+w ' F]); unix(['chmod o-w ' F]); end

ht_updates = numel(update_iters);
myLogInfo('%d Hash Table updates, bits computed: %g', ht_updates, bits_computed_all);
myLogInfo('[T%02d] Saved: %s\n', trialNo, F);
end
