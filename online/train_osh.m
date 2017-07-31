function [train_time, update_time, res_time, ht_updates, bits_computed_all, bitflips] = ...
    train_osh(Xtrain, Ytrain, thr_dist, prefix, test_iters, trialNo, opts)
% Training routine for OSH method, see demo_osh.m .
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
%  bitflips    - (int) total number of bit flips, see update_hash_table.m 
% 
% NOTES
% 	W is d x b where d is the dimensionality 
%            and b is the bit length / # hash functions

%%%%%%%%%%%%%%%%%%%%%%% INIT %%%%%%%%%%%%%%%%%%%%%%%
[W, H, ECOCs] = init_osh(Xtrain, opts);

% NOTE: W_lastupdate keeps track of the last W used to update the hash table
%       W_lastupdate is NOT the W from last iteration
W_lastupdate = W;

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
end

% order training examples
if opts.pObserve > 0
    % [OPTIONAL] order training points according to label arrival strategy
    train_ind = get_ordering(trialNo, Ytrain, opts);
else
    train_ind = zeros(1, opts.epoch*opts.noTrainingPoints);
    for e = 1:opts.epoch
	    % randomly shuffle training points before taking first noTrainingPoints
	    train_ind((e-1)*opts.noTrainingPoints+1:e*opts.noTrainingPoints) = ...
		randperm(size(Xtrain, 1), opts.noTrainingPoints);
    end
end
opts.noTrainingPoints = opts.noTrainingPoints*opts.epoch;
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
res_time    = 0;

% for display
num_labeled   = 0; 
num_unlabeled = 0;

%%%%%%%%%%%%%%%%%%%%%%% STREAMING BEGINS! %%%%%%%%%%%%%%%%%%%%%%%
for iter = 1:opts.noTrainingPoints
    t_ = tic;
    % new training point
    ind = train_ind(iter);
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
    else
        % unlabeled
        isLabeled = false;
        slabel = zeros(size(slabel));  % mark as unlabeled for subsequent functions
        num_unlabeled = num_unlabeled + 1;
    end

    % ---- hash function update ----
    % SGD. update W wrt. loss term(s)
    if isLabeled
        for c = 1:size(target_codes, 1)
            code = target_codes(c, :);
            W = sgd_update(W, spoint, code, opts.stepsize, opts.SGDBoost);
        end
    end
    
    train_time = train_time + toc(t_);
    
    % ---- reservoir update & compute new reservoir hash table ----
    t_ = tic;
    Hres_new = [];
    if reservoir_size > 0
        [reservoir, update_ind] = update_reservoir(reservoir, ...
            spoint, slabel, reservoir_size, W_lastupdate);
        % compute new reservoir hash table (do not update yet)
        Hres_new = (reservoir.X * W > 0);
    end

    % ---- determine whether to update or not ----
    [update_table, trigger_val, h_ind] = trigger_update(iter, ...
        opts, W_lastupdate, W, reservoir, Hres_new);
    res_time = res_time + toc(t_);
    
    % ---- hash table update, etc ----
    if update_table
        h_ind_array = [h_ind_array; single(ismember(1:opts.nbits, h_ind))];
        W_lastupdate(:, h_ind) = W(:, h_ind);  % W_lastupdate: last W used to update hash table
        update_iters = [update_iters, iter];

        % update reservoir hash table
        if reservoir_size > 0
            reservoir.H = Hres_new;
            if strcmpi(opts.trigger,'bf')
                bitflips_res = bitflips_res + trigger_val;
            end
        end

        % update actual hash table
        t_ = tic;
        [H, bf_all, bits_computed] = update_hash_table(H, W_lastupdate, ...
            Xtrain, Ytrain, h_ind, update_iters, opts, ...
                multi_labeled, seenLabels, M_ecoc);
        bits_computed_all = bits_computed_all + bits_computed;
        bitflips = bitflips + bf_all;
        update_time = update_time + toc(t_);
        
        myLogInfo('[T%02d] HT Update#%d @%d, #BRs=%g, bf_all=%g, trigger_val=%g(%s)', ...
            trialNo, numel(update_iters), iter, bits_computed_all , bf_all, trigger_val, opts.trigger);
    end
    
    % ---- cache intermediate model to disk ----
    % CHECKPOINT
    if ismember(iter, test_iters)
        F = sprintf('%s_iter%d.mat', prefix, iter);
        save(F, 'W', 'W_lastupdate', 'H', 'bitflips','bits_computed_all', ...
            'train_time', 'update_time', 'res_time', 'seenLabels', 'update_iters');

        myLogInfo(['*checkpoint*\n[T%02d] %s\n' ...
            '     (%d/%d) W %.2fs, HT %.2fs(%d updates), Res %.2fs\n' ...
            '     total #BRs=%g, avg #BF=%g'], ...
            trialNo, opts.identifier, iter, opts.noTrainingPoints, ...
            train_time, update_time, numel(update_iters), res_time, ...
            bits_computed_all, bitflips);
    end
end % end for iter
%%%%%%%%%%%%%%%%%%%%%%% STREAMING ENDED! %%%%%%%%%%%%%%%%%%%%%%%

% save final model, etc
F = [prefix '.mat'];
save(F, 'W', 'H', 'bitflips', 'bits_computed_all', ...
    'train_time', 'update_time', 'res_time', 'test_iters', 'update_iters', ...
    'seenLabels', 'h_ind_array');

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
            W(:,id) = W(:,id) + stepsizes(i)*(repmat(xi',[1 n])*diag(ci(id)));%*diag(bal(id));
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

d = size(Xtrain, 2);
if 0
    W = rand(d, opts.nbits)-0.5;
else
    % LSH init
    W = randn(d, opts.nbits);
    W = W ./ repmat(diag(sqrt(W'*W))',d,1);
end
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
