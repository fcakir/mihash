function train_osh(run_trial, opts)
	% online (semi-)supervised hashing

	global Xtrain Ytrain
	train_time  = zeros(1, opts.ntrials);
	update_time = zeros(1, opts.ntrials);
	bit_flips   = zeros(1, opts.ntrials);
	for t = 1:opts.ntrials
		if run_trial(t) == 0
			myLogInfo('Trial %02d not required, skipped', t);
			continue;
		end
		myLogInfo('%s: %d trainPts, random trial %d', opts.identifier, opts.noTrainingPoints, t);

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
		[train_time(t), update_time(t), bit_flips(t)] = sgd_optim(Xtrain, Ytrain, ...
			prefix, test_iters, t, opts);
	end

	myLogInfo('Training time (total): %.2f +/- %.2f', mean(train_time), std(train_time));
	if strcmp(opts.mapping, 'smooth')
		myLogInfo('      Bit flips (per): %.4g +/- %.4g', mean(bit_flips), std(bit_flips));
	end
end


% -------------------------------------------------------------
function [train_time, update_time, bitflips] = sgd_optim(Xtrain, Ytrain, ...
		prefix, test_iters, trialNo, opts)
	% optimization via SGD

	% init
	[W, H, ECOCs] = init_osh(Xtrain, opts);

	% NOTE: W_lastupdate keeps track of the last W used to update the hash table
	%       W_lastupdate is NOT the W from last iteration
	W_lastupdate = W;  

	ntrain_all    = size(Xtrain, 1);
	bitflips      = 0;   bitflips_res = 0;
	train_time    = 0;   update_time  = 0;
	maxLabelSize  = 205; % Sun
	numLabels     = numel(unique(Ytrain));

	debug = 0;
	if debug  % DEBUG: keep reservoir fixed
		ind = randperm(ntrain_all);
		Xsample = Xtrain(ind(1:opts.reservoirSize),:);
		Ysample = Ytrain(ind(1:opts.reservoirSize));
		clear ind;
	end

	% are we handling a mult-labeled dataset?
	multi_labeled = (size(Ytrain, 2) > 1);
	if multi_labeled, myLogInfo('Handling multi-labeled dataset'); end

	% deal with regularizers
	if opts.reg_rs > 0
		% use reservoir sampling regularizer
		reservoir_size = opts.reservoirSize;
		if ~debug
		Xsample        = zeros(reservoir_size, size(Xtrain, 2));
		Ysample        = zeros(reservoir_size, 1);
		end
		priority_queue = zeros(1, reservoir_size);
		Hres           = [];  % mapped binary codes for the reservoir
		% for adaptive threshold
		if opts.adaptive > 0
			persistent table_thr;
			table_thr = arrayfun(@bit_fp_thr, opts.nbits*ones(1,maxLabelSize), ...
				1:maxLabelSize);
		end
	end
	%{
	if opts.reg_maxent > 0
		% use max entropy regularizer
		num_unlabeled = 0;
		U = zeros(size(Xtrain, 2));
	end
	%}

	% SGD iterations
	i_ecoc = 1;  M_ecoc = [];  seenLabels = [];
	update_iters = []; % keep track of when the hash table updates happen
	num_labeled = 0;
	num_unlabeled = 0;

	% [OPTIONAL] order training points according to label arrival strategy
	if opts.pObserve > 0
		train_ind = get_ordering(trialNo, Ytrain, opts);
	else
		train_ind = 1:opts.noTrainingPoints;
	end
	ret_val = 0;

	% STREAMING BEGINS...
	for i = 1:opts.noTrainingPoints
		t_ = tic;  
		% new training point
		ind = train_ind(i);
		spoint = Xtrain(ind, :);
		slabel = Ytrain(ind, :);        

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		% Assign ECOC, etc
		%
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
			if opts.reg_smooth > 0 && opts.reg_rs > 0
				% hack: for the reservoir, smooth mapping is assumed
				if i > reservoir_size
					resY = 2*single(W'*Xsample' > 0)-1;
					qY = 2* single(W'*spoint' > 0)-1;
					[~, ind] = sort(resY' * qY,'descend');
				end
			end
		else  
			% unlabeled
			isLabeled = false;
			slabel = zeros(size(slabel));  % mark as unlabeled for subsequent functions
			num_unlabeled = num_unlabeled + 1;
			%{
			if opts.reg_maxent > 0  % update maxent regularizer
				U = U*num_unlabeled + spoint'*spoint;
				U = U/num_unlabeled;
			end
			%}
		end

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		% hash function update
		%
		% SGD-1. update W wrt. loss term(s)
		if isLabeled
			for c = 1:size(target_codes, 1)
				code = target_codes(c, :);
				W = sgd_update(W, spoint, code, opts.stepsize, opts.SGDBoost);
			end
		end

		% SGD-2. update W wrt. reservoir regularizer (if specified)
		% TODO when to use rs.reg.?
		if isLabeled && opts.reg_rs > 0  &&  i > reservoir_size

			stepsizes = ones(reservoir_size,1)*opts.reg_rs*opts.stepsize/reservoir_size;
			ind = randperm(size(Xsample,1));
			W = sgd_update(W, Xsample(ind(1:opts.sampleResSize),:), Hres(ind(1:opts.sampleResSize),:), ...
				stepsizes(ind(1:opts.sampleResSize)), opts.SGDBoost);
		end

		% SGD-3. update W wrt. unsupervised regularizer (if specified)
		if opts.reg_smooth > 0 && i > reservoir_size && isLabeled
			W = reg_smooth(W, ...
				[spoint; Xsample(ind(1:opts.rs_sm_neigh_size),:)], ...
				opts.reg_smooth);
		%{
		elseif isLabeled && opts.reg_maxent > 0  &&  num_unlabeled > 10
			% TODO hard-coded starting threshold of 10 unlabeled examples
			W = W - opts.reg_maxent * U * W;
		%}
		end
		train_time = train_time + toc(t_);


		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		% reservoir update & compute new reservoir hash table
		%
		if opts.reg_rs > 0
			[Xsample, Ysample, priority_queue, ind] = update_reservoir(...  
				Xsample, Ysample, priority_queue, spoint, slabel, i, reservoir_size);
			
			% compute new reservoir hash table (do not update yet)
			% a hack: we always use smooth mapping for reservoir samples 
			Hres_new = (W' * Xsample' > 0)';

			% NOTE: the old reservoir hash table needs updating too
			%       since Xsample has possibly changed.
			if isempty(Hres)
				Hres = (W_lastupdate' * Xsample' > 0)';
			elseif (ind > 0)
				Hres(ind, :) = (W_lastupdate' * Xsample(ind, :)' > 0)';
			end
		end


		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		% hash index update
		%
		% determine whether to update or not
		[update_table, bf_res] = trigger_update(i, opts, ...
			W_lastupdate, W, Xsample, Ysample, Hres, Hres_new);

		if update_table
			W_lastupdate = W;  % W_lastupdate: last W used to update hash table
			update_iters = [update_iters, i];

			% update reservoir hash table
			Hres = Hres_new;
			bitflips_res = bitflips_res + bf_res;

			% update actual hash table
			t_ = tic;
			[H, bf_all] = update_hash_table(H, W, Xtrain, Ytrain, ...
				multi_labeled, seenLabels, M_ecoc, opts);

			bitflips = bitflips + bf_all;
			update_time = update_time + toc(t_);

			myLogInfo('[T%02d] HT Update#%d @%d, bf_all=%g, bf_res=%g', ...
				trialNo, numel(update_iters), i, bf_all, bf_res);
		end

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		% cache intermediate model to disk
		%
		if ismember(i, test_iters)
			F = sprintf('%s_iter%d.mat', prefix, i);
			save(F, 'W', 'H', 'bitflips', 'train_time', 'update_time', ...
				'seenLabels', 'update_iters');
			if ~opts.windows, unix(['chmod o-w ' F]); end  % matlab permission bug

			myLogInfo(['[T%02d] %s\n' ...
				'            (%d/%d)  SGD %.2fs, HTU %.2fs, %d Updates\n' ...
				'            L=%d, UL=%d, SeenLabels=%d, #BF=%g\n'], ...
				trialNo, opts.identifier, i, opts.noTrainingPoints, ...
				train_time, update_time, numel(update_iters), ...
				num_labeled, num_unlabeled, sum(seenLabels>0), bitflips);
		end
	end % end for

	% save final model, etc
	F = [prefix '.mat'];
	save(F, 'W', 'H', 'bitflips', 'train_time', 'update_time', 'test_iters', ...
		'update_iters','seenLabels');
	if ~opts.windows, unix(['chmod o-w ' F]); end % matlab permission bug

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
% do actual hash table update
function [Hnew, bitflips] = update_hash_table(H, W, Xtrain, Ytrain, ...
		multi_labeled, seenLabels, M_ecoc, opts)

	% recover true labels for single-label case
	if ~multi_labeled, Ytrain = floor(Ytrain/10); end

	% build new table
	if opts.tstScenario == 1
		Hnew = build_hash_table(W, Xtrain, Ytrain, seenLabels, M_ecoc, opts);
	else
		% TODO: what does it mean?
		error('not implemented yet');
		%Hnew = build_hash_table(W, Xtrain(1:i,:), Ytrain(1:i,:), seenLabels, M_ecoc, opts);
	end

	% compute bitflips
	if isempty(H)
		bitflips = 0;
	else
		if opts.tstScenario == 2
			% TODO: what does it mean?
			error('not implemented yet');
			%bitdiff = xor(H, Hnew(:, 1:update_iters(end-1)));
		else
			bitdiff = xor(H, Hnew);
		end
		bitflips = sum(bitdiff(:))/size(Xtrain, 1);
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
% reservoir sampling, update step, based on random sort
function [Xsample, Ysample, priority_queue, ind] = update_reservoir(...
		Xsample, Ysample, priority_queue, spoint, slabel, i, reservoir_size)
	% outputs: 
	%   Xsample, Ysample, priority_queue: updated
	%   ind: updated index (0 for no update)
	if i <= reservoir_size
		Xsample(i, :)     = spoint;
		Ysample(i)        = slabel;
		priority_queue(i) = rand;
		ind = i;
	else
		% pop max from priority queue
		[maxval, maxind] = max(priority_queue);
		r = rand;
		if maxval > r
			% push into priority queue
			priority_queue(maxind) = r;
			Xsample(maxind, :)     = spoint;
			Ysample(maxind)        = slabel;
			ind = maxind;
		else
			ind = 0;  % no update
		end
	end
end

% -----------------------------------------------------------
% smoothness regularizer
function W = reg_smooth(W, points, reg_smooth)
	reg_smooth = reg_smooth/size(points,1);
	% try
	for i = 1:size(W,2)
		gradWi = zeros(size(W,1),1);
		for j = 2:size(points,1)
			gradWi = gradWi + points(1,:)'*(W(:,i)'*points(j,:)') + ...
				(W(:,i)'*points(1,:)')*points(j,:)';
		end
		W(:,i) = W(:,i) - reg_smooth * gradWi;
	end
	%catch e
	%    disp(e.message);
	%    keyboard
	%end
end
