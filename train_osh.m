function [train_time, update_time, res_time, ht_updates, bits_computed_all, bitflips] = ...
    train_osh(Xtrain, Ytrain, thr_dist, prefix, test_iters, trialNo, opts)

%%%%%%%%%%%%%%%%%%%%%%% INIT %%%%%%%%%%%%%%%%%%%%%%%
if ~opts.learn_ecoc
	[W, H, ECOCs] = init_osh(Xtrain, opts);
	uYtrain = [];c_idx = [];c_centers = [];b_w = [];
else
	[W, H, ECOCs, uYtrain, c_centers, c_idx, b_w] = init_osh_l(Xtrain, Ytrain, opts);
end
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
    %if opts.adaptive > 0
    %    maxLabelSize = 205; % Sun
    %    persistent adaptive_thr;
    %    adaptive_thr = arrayfun(@bit_fp_thr, opts.nbits*ones(1,maxLabelSize), ...
    %        1:maxLabelSize);
    %end
end

% order training examples
if opts.pObserve > 0
    % [OPTIONAL] order training points according to label arrival strategy
    train_ind = get_ordering(trialNo, Ytrain, opts);
else
    train_ind = zeros(1, opts.epoch*opts.noTrainingPoints);
    % For nus, the classes (combinations of labels) are highly unbalanced -this shuffles data so that 
    % train_ind contains data from all possible combinations of labels with equal probability
    if opts.learn_ecoc && strcmp(opts.dataset,'nus')
	    % ToDo: currently only works without clustering, i.e., c_centers represent all uYtrain
	    [uY, ia, ic] = unique(Ytrain, 'rows');
	    count = 0;
	    i_ = 1:length(ic);
	    ic_ = ic;
	    while count < opts.noTrainingPoints*opts.epoch
	    	[~,loc] = ismember(1:length(ia), ic_);
		inds = i_(loc);
		train_ind(count+1:count+length(loc)) = inds(randperm(length(inds)));
		assert(size(unique(Ytrain(i_(loc), :),'rows'),1) == length(ia));
		i_ = randperm(length(ic));
		ic_ = ic(i_);
		count = count + length(loc);
	    end
	    %train_ind = train_ind(randperm(length(train_ind)));
    else
	    for e = 1:opts.epoch
		% randomly shuffle training points before taking first noTrainingPoints
		train_ind((e-1)*opts.noTrainingPoints+1:e*opts.noTrainingPoints) = ...
			randperm(size(Xtrain, 1), opts.noTrainingPoints);
	    end
    end
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
res_time    = 0;

% for display
num_labeled   = 0; 
num_unlabeled = 0;
%%%%%%%%%%%%%%%%%%%%%%% SET UP OSH %%%%%%%%%%%%%%%%%%%%%%%
opts.noTrainingPoints = opts.noTrainingPoints*opts.epoch;

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
            slabel, seenLabels, M_ecoc, i_ecoc, ECOCs, opts.learn_ecoc, uYtrain, c_centers, c_idx);
        
        % if using smoothness regularizer:
        % When a labelled items comes find its neighors from the reservoir
        if opts.reg_smooth > 0 && reservoir_size > 0
            % hack: for the reservoir, smooth mapping is assumed
            if iter > reservoir_size
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
            W = sgd_update(W, spoint, code, opts.stepsize, opts.SGDBoost, b_w);
        end
    end
    
    % SGD-2. update W wrt. reservoir regularizer (if specified)
    if (isLabeled) && (opts.reg_rs>0) && (iter>reservoir_size)
        stepsizes = ones(reservoir_size,1) / reservoir_size;
        stepsizes = stepsizes * opts.stepsize * opts.reg_rs;
        ind = randperm(reservoir.size, opts.sampleResSize);
        W = sgd_update(W, reservoir.X(ind,:), reservoir.H(ind,:), ...
            stepsizes(ind), opts.SGDBoost);
    end
    
    % SGD-3. update W wrt. unsupervised regularizer (if specified)
    if opts.reg_smooth > 0 && iter > reservoir_size && isLabeled
        ind = randperm(reservoir.size, opts.rs_sm_neigh_size);
        W = reg_smooth(W, [spoint; reservoir.X(ind,:)], opts.reg_smooth);
    end
    
    % % SGD-4. apply accumulated gradients (if applicable)
    % if reservoir_size > 0 && opts.accuHash > 0
    %     W = W - stepW;
    %     stepW = zeros(size(W));
    % end
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
    %
    % [DEPRECATED: adaptive]
    %if opts.adaptive > 0
    %    bf_thr = adaptive_thr(max(1, length(seenLabels)));
    %    [update_table, trigger_val] = trigger_update(iter, ...
    %        opts, W_lastupdate, W, reservoir, Hres_new);
    %    h_ind = 1:opts.nbits;
    %    inv_h_ind = [];
    %else
    [update_table, trigger_val, h_ind] = trigger_update(iter, ...
        opts, W_lastupdate, W, reservoir, Hres_new);
    inv_h_ind = setdiff(1:opts.nbits, h_ind);  % keep these bits unchanged
    if reservoir_size > 0 && numel(h_ind) < opts.nbits  % selective update
        %assert(opts.fracHash < 1);
        Hres_new(:, inv_h_ind) = reservoir.H(:, inv_h_ind);
    end
    %end
    res_time = res_time + toc(t_);
    
    % ---- hash table update, etc ----
    if update_table
        h_ind_array = [h_ind_array; single(ismember(1:opts.nbits, h_ind))];
        W_lastupdate(:, h_ind) = W(:, h_ind);  % W_lastupdate: last W used to update hash table
        if opts.accuHash <= 0
            % no gradient accumulation: 
            %   throw away increments in unused hash bits
            W = W_lastupdate;
            myLogInfo('not accumulating gradients!');
        end
        if opts.fracHash < 1
            myLogInfo('selective update: fracHash=%g, randomHash=%g', ...
                opts.fracHash, opts.randomHash);
        end
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
    %
    if ismember(iter, test_iters)
        F = sprintf('%s_iter%d.mat', prefix, iter);
        save(F, 'W', 'W_lastupdate', 'H', 'bitflips','bits_computed_all', ...
            'train_time', 'update_time', 'res_time', 'seenLabels', 'update_iters');
        % fix permission
        if ~opts.windows, unix(['chmod g+w ' F]); unix(['chmod o-w ' F]); end

        myLogInfo(['[T%02d] %s\n' ...
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
% fix permission
if ~opts.windows, unix(['chmod g+w ' F]); unix(['chmod o-w ' F]); end

ht_updates = numel(update_iters);
myLogInfo('%d Hash Table updates, bits computed: %g', ht_updates, bits_computed_all);
myLogInfo('[T%02d] Saved: %s\n', trialNo, F);
end

% -----------------------------------------------------------
% SGD mini-batch update
function W = sgd_update(W, points, codes, stepsizes, SGDBoost, b_w)
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
        %bal = zeros(1, length(b_w));
        xi = points(i, :);
        ci = codes(i, :);
        bal(ci == 1) = 1 - b_w(ci == 1);
        bal(ci == -1) = b_w(ci == -1);
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
% initialize online hashing
function [W, H, ECOCs, uYtrain, c_centers, c_idx, b_w] = init_osh_l(Xtrain, Ytrain, opts)
% randomly generate candidate codewords, store in ECOCs
[uYtrain] = unique(Ytrain, 'rows');
c_centers = []; c_idx = [];
if ~strcmpi(opts.dataset, 'nus')
	% generally speaking, the number of classes in multiclass datasets are <= 1K
	S = 2*(repmat(uYtrain,1,length(uYtrain)) == repmat(uYtrain,1,length(uYtrain))') - 1;
else
	% if number of distinct label combinations is large, then we need to cluster them
	% otherwise there will be only a few instances to train for each of the combinations
	K = size(uYtrain, 1);
	if K > opts.cluster_size
		myLogInfo(sprintf('Too many (%d) combinations, clustering...', K));
		[c_idx, c_centers] = kmedoids(uYtrain, opts.cluster_size, 'Distance', 'jaccard');
		u_c_idx = unique(c_idx);
		D = zeros(length(u_c_idx), length(u_c_idx));
	        D(1:length(u_c_idx)+1:length(u_c_idx)^2) = 1;
		for i = 1:length(u_c_idx)-1
			for j = i+1:length(u_c_idx)
				ci_ = find(c_idx == u_c_idx(i));
				cj_ = find(c_idx == u_c_idx(j));
				D(i, j) = sum(sum(pdist2(uYtrain(ci_, :), uYtrain(cj_, :), 'jaccard')))/(length(ci_)*length(cj_));
				D(j, i) = D(i, j);
			end
		end
    
		D = D - min(D(:));
	        D = D ./ max(D(:));
	        myLogInfo('MAX D=%g, MIN D=%g', max(D(:)), min(D(:)));
		S = 2*D - 1;
	else
		c_idx = 1:K;
		c_centers = uYtrain;
		S = 2*single(c_centers*c_centers' > 0 ) - 1;
	end
	
	%K = randperm(K);
	%uYtrain = uYtrain(K(1:5000),:);
	% even though you're using k-medoids the below can give you sparse S
	% mean of each cluster > 1/2 would be more appropiate
	%S = 2*single(c_centers * c_centers' > 0) - 1;
end
S = S * opts.nbits;
bigM = size(S, 1);

% NOTE ECOCs now is a BINARY (0/1) MATRIX!
ECOCs = logical(zeros(bigM, opts.nbits));

if 1 %strcmpi(opts.dataset,'nus')
	for t = 1:opts.nbits
		if t > 1
			y = 2*single(y) - 1;
			S = S - y*y';
		end
		[U, V] = eig(single(S));
		eigenvalue = diag(V)';
		[eigenvalue, order] = sort(eigenvalue, 'descend');
		y = U(:, order(1));
		y = y > 0;
		ECOCs(:, t) = y;
	end
else
% keep below block for debugging purposes
	for t = 1:opts.nbits
	    r = ones(bigM, 1);
	    while (sum(r)==bigM || sum(r)==0)
		r = randi([0,1], bigM, 1);
	    end
	    ECOCs(:, t) = logical(r);
	end
end
% arranging stepsize for unbalanced class distribution
b_w = (mean(2*ECOCs-1,1)+1)./2;
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
    slabel, seenLabels, M_ecoc, i_ecoc, ECOCs, l_ecoc, uYtrain, c_centers, c_idx)
assert(sum(slabel) ~= 0, 'Error: finding target codes for unlabeled example');

if numel(slabel) == 1
    % single-label dataset
    [ismem, ind] = ismember(slabel, seenLabels);
    if ismem == 0
        seenLabels = [seenLabels; slabel];
        % NOTE ECOCs now is a BINARY (0/1) MATRIX!
	if ~l_ecoc
	        M_ecoc = [M_ecoc; 2*ECOCs(i_ecoc,:)-1];
	else
		ind = find(uYtrain == slabel*10);
		M_ecoc = [M_ecoc; 2*ECOCs(ind, :) - 1];
	end
        ind    = i_ecoc;
        i_ecoc = i_ecoc + 1;
    end
    
else
    if ~l_ecoc
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
    else
        if isempty(seenLabels)
            assert(isempty(M_ecoc));
            seenLabels = zeros(1, length(c_idx));
            M_ecoc = 2*ECOCs-1;
        end
        
        %if isempty(M_ecoc), M_ecoc = 2*ECOCs - 1; end;
        ind = find(ismember(uYtrain, slabel, 'rows'));        
        assert(~isempty(ind));
        seenLabels(ind) = 1;
        
    end
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
