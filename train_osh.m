function train_osh(run_trial, opts)
% online (semi-)supervised hashing

global Xtrain Ytrain
train_time  = zeros(1, opts.ntrials);
update_time = zeros(1, opts.ntrials);
ht_updates  = zeros(1, opts.ntrials);
bit_flips   = zeros(1, opts.ntrials);
bit_recomp  = zeros(1, opts.ntrials);
for t = 1:opts.ntrials
    if run_trial(t) == 0
        myLogInfo('Trial %02d not required, skipped', t);
        continue;
    end
    myLogInfo('%s: %d trainPts, random trial %d', opts.identifier, opts.noTrainingPoints, t);
    
    % % shuffle training data for each trial
    % ind    = randperm(length(Ytrain));
    % Ytrain_ = Ytrain(ind);
    % Xtrain_ = Xtrain(ind, :);
    
    % randomly set test checkpoints (to better mimic real scenarios)
    test_iters      = zeros(1, opts.ntests);
    test_iters(1)   = 1;
    test_iters(end) = opts.noTrainingPoints;
    interval = round(opts.noTrainingPoints/(opts.ntests-1));
    for i = 1:opts.ntests-2
        iter = interval*i + randi([1 round(interval/3)]) - round(interval/6);
        test_iters(i+1) = iter;
    end
    prefix = sprintf('%s/trial%d', opts.expdir, t);
    
    % do SGD optimization
    [train_time(t), update_time(t), ht_updates(t), bit_recomp(t), bit_flips(t)] ...
        = OSH(Xtrain, Ytrain, prefix, test_iters, t, opts);
end

myLogInfo('Training time (total): %.2f +/- %.2f', mean(train_time), std(train_time));
myLogInfo('HTupdate time (total): %.2f +/- %.2f', mean(update_time), std(update_time));
if strcmp(opts.mapping, 'smooth')
    %TODO
    myLogInfo('    Hash Table Updates (per): %.4g +/- %.4g', mean(ht_updates), std(ht_updates));
    myLogInfo('    Bit Recomputations (per): %.4g +/- %.4g', mean(bit_recomp), std(bit_recomp));
    myLogInfo('    Bit flips (per): %.4g +/- %.4g', mean(bit_flips), std(bit_flips));
end
end


% -------------------------------------------------------------
function [train_time, update_time, ht_updates, bits_computed_all, bitflips] = ...
    OSH(Xtrain, Ytrain, prefix, test_iters, trialNo, opts)
% optimization via SGD

%%%%%%%%%%%%%%%%%%%%%%% INIT %%%%%%%%%%%%%%%%%%%%%%%
[W, H, ECOCs] = init_osh(Xtrain, opts);
% NOTE: W_lastupdate keeps track of the last W used to update the hash table
%       W_lastupdate is NOT the W from last iteration
W_lastupdate = W;
stepW = zeros(size(W));  % Gradient accumulation matrix

% are we handling a mult-labeled dataset?
multi_labeled = (size(Ytrain, 2) > 1);
if multi_labeled, myLogInfo('Handling multi-labeled dataset'); end

% set up reservoir
reservoir = [];
reservoir_size = opts.reservoirSize;
if reservoir_size > 0
    reservoir.size = 0;
    reservoir.X    = zeros(0, size(Xtrain, 2));
    reservoir.Y    = zeros(0, size(Ytrain, 2));
    reservoir.PQ   = [];
    reservoir.H    = [];  % mapped binary codes for the reservoir
    
    % for adaptive threshold
    if opts.adaptive > 0
        maxLabelSize = 205; % Sun
        persistent adaptive_thr;
        adaptive_thr = arrayfun(@bit_fp_thr, opts.nbits*ones(1,maxLabelSize), ...
            1:maxLabelSize);
    end
end

% order training examples
if opts.pObserve > 0
    % [OPTIONAL] order training points according to label arrival strategy
    train_ind = get_ordering(trialNo, Ytrain, opts);
else
    % randomly shuffle training points before taking first noTrainingPoints
    % this fixes issue #25
    train_ind = randperm(size(Xtrain, 1), opts.noTrainingPoints);
end
%%%%%%%%%%%%%%%%%%%%%%% INIT %%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%% SET UP OSH %%%%%%%%%%%%%%%%%%%%%%%
% for ECOC
i_ecoc     = 1;  
M_ecoc     = [];  
seenLabels = [];

% bit flips & bits computed
bitflips          = 0;
bitflips_res      = 0;
bits_computed_all = 0;

% HT updates
update_iters = [];
h_ind_array  = [];

% for recording time
train_time  = 0;  
update_time = 0;

% for display
num_labeled   = 0; 
num_unlabeled = 0;
%%%%%%%%%%%%%%%%%%%%%%% SET UP OSH %%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%% STREAMING BEGINS! %%%%%%%%%%%%%%%%%%%%%%%
for i = 1:opts.noTrainingPoints
    t_ = tic;
    % new training point
    ind = train_ind(i);
    spoint = Xtrain(ind, :);
    slabel = Ytrain(ind, :);
    
    % ---- Assign ECOC, etc ----
    if (~multi_labeled && mod(slabel, 10) == 0) || ...
            (multi_labeled && sum(slabel) > 0)
        % labeled (single- or multi-label): assign target code(s)
        isLabeled = true;
        if ~multi_labeled
            slabel = slabel/10;  % single-label: recover true label in [1, L]
        end
        num_labeled = num_labeled + 1;
        [target_codes, seenLabels, M_ecoc, i_ecoc] = find_target_codes(...
            slabel, seenLabels, M_ecoc, i_ecoc, ECOCs);
        
        % if using smoothness regularizer:
        % When a labelled items comes find its neighors from the reservoir
        if opts.reg_smooth > 0 && reservoir_size > 0
            % hack: for the reservoir, smooth mapping is assumed
            if i > reservoir_size
                resY = 2*single(W'*reservoir.X' > 0)-1;
                qY = 2* single(W'*spoint' > 0)-1;
                [~, ind] = sort(resY' * qY,'descend');
            end
        end
    else
        % unlabeled
        isLabeled = false;
        slabel = zeros(size(slabel));  % mark as unlabeled for subsequent functions
        num_unlabeled = num_unlabeled + 1;
    end
    
    % ---- hash function update ----
    % SGD-1. update W wrt. loss term(s)
    if isLabeled
        for c = 1:size(target_codes, 1)
            code = target_codes(c, :);
            W = sgd_update(W, spoint, code, opts.stepsize, opts.SGDBoost);
        end
    end
    
    % SGD-2. update W wrt. reservoir regularizer (if specified)
    if (isLabeled) && (opts.reg_rs>0) && (i>reservoir_size)
        stepsizes = ones(reservoir_size,1) / reservoir_size;
        stepsizes = stepsizes * opts.stepsize * opts.reg_rs;
        ind = randperm(reservoir.size, opts.sampleResSize);
        W = sgd_update(W, reservoir.X(ind,:), reservoir.H(ind,:), ...
            stepsizes(ind), opts.SGDBoost);
    end
    
    % SGD-3. update W wrt. unsupervised regularizer (if specified)
    if opts.reg_smooth > 0 && i > reservoir_size && isLabeled
        ind = randperm(reservoir.size, opts.rs_sm_neigh_size);
        W = reg_smooth(W, [spoint; reservoir.X(ind,:)], opts.reg_smooth);
    end
    
    % SGD-4. apply accumulated gradients (if applicable)
    if reservoir_size > 0 && opts.accuHash > 0
        W = W - stepW;
        stepW = zeros(size(W));
    end
    train_time = train_time + toc(t_);
    
    % ---- reservoir update & compute new reservoir hash table ----
    if reservoir_size > 0
        [reservoir, update_ind] = update_reservoir(reservoir, ...
            spoint, slabel, reservoir_size, W_lastupdate);
        % compute new reservoir hash table (do not update yet)
        Hres_new = (reservoir.X * W > 0);
    end

    % ---- determine whether to update or not ----
    if opts.adaptive > 0
        bf_thr = adaptive_thr(max(1, length(seenLabels)));
        [update_table, trigger_val] = trigger_update(i, opts, ...
            W_lastupdate, W, reservoir, Hres_new, bf_thr);
        h_ind = 1:opts.nbits;
        inv_h_ind = [];
    else
        [update_table, trigger_val, h_ind] = trigger_update(i, opts, ...
            W_lastupdate, W, reservoir, Hres_new);
        inv_h_ind = setdiff(1:opts.nbits, h_ind);  % keep these bits unchanged
        if reservoir_size > 0 && numel(h_ind) < opts.nbits  % selective update
            assert(opts.fracHash < 1);
            Hres_new(:, inv_h_ind) = reservoir.H(:, inv_h_ind);
        end
    end
    
    % ---- hash table update, etc ----
    if update_table
	h_ind_array = [h_ind_array ; single(ismember(1:opts.nbits, h_ind))];

	W_lastupdate(:, h_ind) = W(:, h_ind);  % W_lastupdate: last W used to update hash table
        if opts.accuHash > 0 && ~isempty(inv_h_ind)
            assert(sum(sum(abs((W_lastupdate - stepW) - W))) < 1e-10);
            stepW(:, inv_h_ind) = W_lastupdate(:, inv_h_ind) - W(:, inv_h_ind);
        end
        W = W_lastupdate;
        update_iters = [update_iters, i];

        % update reservoir hash table
        if reservoir_size > 0
            reservoir.H = Hres_new;
            if strcmpi(opts.trigger,'bf')
                bitflips_res = bitflips_res + trigger_val;
            end
        end

        % update actual hash table
        t_ = tic;
        [H, bf_all, bits_computed] = update_hash_table(H, W, Xtrain, Ytrain, ...
            h_ind, update_iters, opts, ...
            multi_labeled, seenLabels, M_ecoc);
        bits_computed_all = bits_computed_all + bits_computed;
	bitflips = bitflips + bf_all;
        update_time = update_time + toc(t_);
        
        myLogInfo('[T%02d] HT Update#%d @%d, #BRs=%g, bf_all=%g, trigger_val=%g(%s)', ...
            trialNo, numel(update_iters), i, bits_computed_all , bf_all, trigger_val, opts.trigger);
    end
    
    % ---- cache intermediate model to disk ----
    %
    if ismember(i, test_iters)
        F = sprintf('%s_iter%d.mat', prefix, i);
        save(F, 'W', 'H', 'bitflips','bits_computed_all', 'train_time', 'update_time', ...
            'seenLabels', 'update_iters');
        % fix permission
        if ~opts.windows, unix(['chmod g+w ' F]); unix(['chmod o-w ' F]); end

        myLogInfo(['[T%02d] %s\n' ...
            '            (%d/%d)  SGD %.2fs, HTU %.2fs, %d Updates\n' ...
            '            #BRs=%g, L=%d, UL=%d, SeenLabels=%d, #BF=%g\n'], ...
            trialNo, opts.identifier, i, opts.noTrainingPoints, ...
            train_time, update_time, numel(update_iters), ...
            bits_computed_all, num_labeled, num_unlabeled, sum(seenLabels>0), bitflips);
    end
end % end for i
%%%%%%%%%%%%%%%%%%%%%%% STREAMING ENDED! %%%%%%%%%%%%%%%%%%%%%%%

% save final model, etc
F = [prefix '.mat'];
save(F, 'W', 'H', 'bitflips', 'bits_computed_all', ...
    'train_time', 'update_time', 'test_iters', 'update_iters', ...
    'seenLabels', 'h_ind_array');
% fix permission
if ~opts.windows, unix(['chmod g+w ' F]); unix(['chmod o-w ' F]); end

ht_updates = numel(update_iters);
myLogInfo('%d Hash Table updates, bits computed: %g', ht_updates, bits_computed_all);
myLogInfo('[T%02d] Saved: %s\n', trialNo, F);
end

% -----------------------------------------------------------
% SGD mini-batch update
function W = sgd_update(W, points, codes, stepsizes, SGDBoost)
% input:
%   W         - D*nbits matrix, each col is a hyperplane
%   points    - n*D matrix, each row is a point
%   codes     - n*nbits matrix, each row the corresp. target binary code
%   stepsizes - SGD step sizes (1 per point) for current batch
% output:
%   updated W
if SGDBoost == 0
    % no online boosting, hinge loss
    for i = 1:size(points, 1)
        xi = points(i, :);
        ci = codes(i, :);
        id = (xi * W .* ci < 1);  % logical indexing > find()
        n  = sum(id);
        if n > 0
            W(:,id) = W(:,id) + stepsizes(i)*repmat(xi',[1 n])*diag(ci(id));
        end
    end
else
    % online boosting + exp loss
    for i = 1:size(points, 1)
        xi = points(i, :);
        ci = codes(i, :);
        st = stepsizes(i);
        for j = 1:size(W, 2)
            if j ~= 1
                c1 = exp(-(ci(1:j-1)*(W(:,1:j-1)'*xi')));
            else
                c1 = 1;
            end
            W(:,j) = W(:,j) - st * c1 * exp(-ci(j)*W(:,j)'*xi')*-ci(j)*xi';
        end
    end
end
end


% -----------------------------------------------------------
% initialize online hashing
function [W, H, ECOCs] = init_osh(Xtrain, opts, bigM)
% randomly generate candidate codewords, store in ECOCs
if nargin < 3, bigM = 10000; end

% NOTE ECOCs now is a BINARY (0/1) MATRIX!
ECOCs = logical(zeros(bigM, opts.nbits));
for t = 1:opts.nbits
    r = ones(bigM, 1);
    while (sum(r)==bigM || sum(r)==0)
        r = randi([0,1], bigM, 1);
    end
    ECOCs(:, t) = logical(r);
end
clear r

% initialize with LSH
d = size(Xtrain, 2);
W = randn(d, opts.nbits);
W = W ./ repmat(diag(sqrt(W'*W))',d,1);
H = [];  % the indexing structure
end

% -----------------------------------------------------------
% find target codes for a new labeled example
function [target_codes, seenLabels, M_ecoc, i_ecoc] = find_target_codes(...
    slabel, seenLabels, M_ecoc, i_ecoc, ECOCs)
assert(sum(slabel) ~= 0, 'Error: finding target codes for unlabeled example');

if numel(slabel) == 1
    % single-label dataset
    [ismem, ind] = ismember(slabel, seenLabels);
    if ismem == 0
        seenLabels = [seenLabels; slabel];
        % NOTE ECOCs now is a BINARY (0/1) MATRIX!
        M_ecoc = [M_ecoc; 2*ECOCs(i_ecoc,:)-1];
        ind    = i_ecoc;
        i_ecoc = i_ecoc + 1;
    end
    
else
    % multi-label dataset
    if isempty(seenLabels)
        assert(isempty(M_ecoc));
        seenLabels = zeros(size(slabel));
        M_ecoc = zeros(numel(slabel), size(ECOCs, 2));
    end
    % find incoming labels that are unseen
    unseen = find((slabel==1) & (seenLabels==0));
    if ~isempty(unseen)
        for j = unseen
            % NOTE ECOCs now is a BINARY (0/1) MATRIX!
            M_ecoc(j, :) = 2*ECOCs(i_ecoc, :)-1;
            i_ecoc = i_ecoc + 1;
        end
        seenLabels(unseen) = 1;
    end
    ind = find(slabel==1);
end

% find/assign target codes
target_codes = M_ecoc(ind, :);
end

% -----------------------------------------------------------
% smoothness regularizer
function W = reg_smooth(W, points, reg_smooth)
reg_smooth = reg_smooth/size(points,1);
for i = 1:size(W,2)
    gradWi = zeros(size(W,1),1);
    for j = 2:size(points,1)
        gradWi = gradWi + points(1,:)'*(W(:,i)'*points(j,:)') + ...
            (W(:,i)'*points(1,:)')*points(j,:)';
    end
    W(:,i) = W(:,i) - reg_smooth * gradWi;
end
end
