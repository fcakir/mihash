function train_osh(traingist, trainlabels, opts)
	% online (semi-)supervised hashing
	
	train_time  = zeros(1, opts.ntrials);
	update_time = zeros(1, opts.ntrials);
	bit_flips   = zeros(1, opts.ntrials);
	parfor t = 1:opts.ntrials
		myLogInfo('%s: random trial %d', opts.identifier, t);
		[train_time(t), update_time(t), bit_flips(t)] = sgd_optim(...
			traingist, trainlabels, opts, t);
	end
	myLogInfo('Training time (total): %.2f +/- %.2f', mean(train_time), std(train_time));
	if strcmp(opts.mapping, 'smooth')
		myLogInfo('      Bit flips (per): %.4g +/- %.4g', mean(bit_flips), std(bit_flips));
	end
end

% -------------------------------------------------------------
function [train_time, update_time, bitflips] = sgd_optim(...
		traingist, trainlabels, opts, trialNo)
	% optimization via SGD
	
	prefix = sprintf('%s/trial%d', opts.expdir, trialNo);
	noexist = 0;
	for i = 1:floor(opts.noTrainingPoints/opts.test_interval)
		if ~exist(sprintf('%s_iter%d.mat', prefix, i), 'file')
			noexist = noexist + 1;
		end
	end
	if noexist == 0 && exist([prefix '.mat'], 'file')
		myLogInfo('Trial %d already done.', trialNo); 
		load([prefix '.mat']); return;
	end

	% init
	[W, Y, ECOCs] = init_osh(traingist, opts);
	ntrain_all    = size(traingist, 1);
	bitflips      = 0;  bitflips_res  = 0;
	train_time    = 0;  update_time   = 0;

	% deal with regularizers
	if opts.reg_rs > 0
		% use reservoir sampling regularizer
		reservoir_size = opts.samplesize; %ceil(opts.sampleratio*ntrain_all);
		samplegist     = zeros(reservoir_size, size(traingist, 2));
		samplelabel    = zeros(reservoir_size, 1);
		priority_queue = zeros(1, reservoir_size);
		Yres           = [];  % mapped binary codes for the reservoir
	end
	if opts.reg_maxent > 0
		% use max entropy regularizer
		num_unlabeled = 0;
		U = zeros(size(traingist, 2));
	end

	% SGD iterations
	i_ecoc = 1;  M_ecoc = [];  seenLabels = [];
	for i = 1:opts.noTrainingPoints
		t_ = tic;  
		% new training point
		spoint = traingist(i, :);
		slabel = trainlabels(i, :);

		if sum(slabel) > 0  % labeled: assign target code(s)
			isLabeled = true;
			[target_codes, seenLabels, M_ecoc, i_ecoc] = find_target_codes(...
				slabel, seenLabels, M_ecoc, i_ecoc, ECOCs);
		else  % unlabeled
			isLabeled = false;
			if opts.reg_maxent > 0  % update maxent regularizer
				U = U*num_unlabeled + spoint'*spoint;
				num_unlabeled = num_unlabeled + 1;
				U = U/num_unlabeled;
			elseif opts.reg_smooth > 0
				; %TODO
			end
		end

		if opts.reg_rs > 0  % reservoir update
			[samplegist, samplelabel, priority_queue] = update_reservoir(...
				samplegist, samplelabel, priority_queue, spoint, slabel, i, reservoir_size);
			Ynew = build_hash_table(W, samplegist, samplelabel, seenLabels, M_ecoc, opts)';
			if ~isempty(Yres)
				bitdiff = (Yres ~= Ynew);
				bitflips_res = bitflips_res + sum(bitdiff(:));
			end
			Yres = Ynew;
		end

		% hash function update
		% SGD-1. update W wrt. loss term(s)
		if isLabeled
			for c = 1:size(target_codes, 1)
				code = target_codes(c, :);
				W = sgd_update(W, spoint, code, opts.stepsize, opts.SGDBoost);
			end
		end
		% SGD-2. update W wrt. reservoir regularizer (if specified)
		if opts.reg_rs > 0  &&  i > reservoir_size
			stepsizes = ones(reservoir_size,1)*opts.reg_rs*opts.stepsize/reservoir_size;
			W = sgd_update(W, samplegist, Yres, stepsizes, opts.SGDBoost);
		end
		% SGD-3. update W wrt. unsupervised regularizer (if specified)
		% either max entropy or smoothness, but not both
		if opts.reg_maxent > 0  &&  num_unlabeled > 10
			W = W - opts.reg_maxent * U * W;
		elseif opts.reg_smooth > 0
			; %TODO
		end
		train_time = train_time + toc(t_);

		% hash index update
		if strcmp(opts.mapping, 'smooth') && ~mod(i, opts.update_interval)
			t_ = tic;
			Ynew = build_hash_table(W, traingist, trainlabels, seenLabels, M_ecoc, opts);
			if ~isempty(Y)
				bitdiff = (Y ~= Ynew);
				bitflips = bitflips + sum(bitdiff(:));
			end
			Y = Ynew;
			update_time = update_time + toc(t_);
		end

		% cache intermediate model to disk
		if ~mod(i, opts.test_interval)
			if isempty(Y)
				Y = build_hash_table(W, traingist, trainlabels, seenLabels, M_ecoc, opts);
			end
			savefile = sprintf('%s_iter%d.mat', prefix, i);
			save(savefile, 'W', 'Y', 'bitflips', 'train_time', 'update_time');
			unix(['chmod o-w ' savefile]);  % matlab permission bug
		end
	end % end for
	bitflips = bitflips/ntrain_all;
	if opts.reg_rs > 0
		bitflips_res = bitflips_res/reservoir_size;
		myLogInfo('Trial %02d. bitflips_res = %g', trialNo, bitflips_res);
	end
	if opts.reg_maxent > 0
		myLogInfo('Trial %02d. %d labeled, %d unlabeled. reg_maxent = %g', ...
			trialNo, opts.noTrainingPoints-num_unlabeled, num_unlabeled, opts.reg_maxent);
	end

	% populate hash table
	t_ = tic;
	Y = build_hash_table(W, traingist, trainlabels, seenLabels, M_ecoc, opts);
	update_time = update_time + toc(t_);
	myLogInfo('Trial %02d. SGD: %.2f sec, Hashtable update: %.2f sec', ...
		trialNo, train_time, update_time);

	% save final model, etc
	save([prefix '.mat'], 'W', 'Y', 'bitflips', 'train_time', 'update_time');
	unix(['chmod o-w ' prefix '.mat']);  % matlab permission bug
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
			id = (xi * W .* ci <= 1);  % logical indexing > find()
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
function [W, Y, ECOCs] = init_osh(traingist, opts, bigM)
	% randomly generate candidate codewords, store in ECOCs
	if nargin < 3, bigM = 10000; end
	ECOCs = zeros(bigM, opts.nbits);
	for t = 1:opts.nbits
		r = ones(bigM, 1);
		while (abs(sum(r)) == bigM)
			r = 2*randi([0,1], bigM, 1)-1;
		end
		ECOCs(:, t) = r;
	end
	clear r

	% initialize with LSH
	d = size(traingist, 2);
	W = randn(d, opts.nbits);
	W = W ./ repmat(diag(sqrt(W'*W))',d,1);
	Y = [];  % the indexing structure
end

% -----------------------------------------------------------
% find target codes for a new labeled example
function [target_codes, seenLabels, M_ecoc, i_ecoc] = find_target_codes(...
		slabel, seenLabels, M_ecoc, i_ecoc, ECOCs);
	assert(sum(slabel) ~= 0, 'Error: finding target codes for unlabeled example');

	if numel(slabel) == 1  % single-label dataset
		ind = find(seenLabels == slabel);
		if isempty(ind)
			if isempty(seenLabels)
				% does not exist, create a binary code for M_ecoc
				seenLabels = slabel;
				M_ecoc = ECOCs(i_ecoc, :);
				i_ecoc = i_ecoc + 1;
			else
				% append codeword to ECOC matrix
				seenLabels = [seenLabels; slabel];
				M_ecoc = [M_ecoc; ECOCs(i_ecoc,:)];
				i_ecoc = i_ecoc +1;
			end
		end
		ind = find(seenLabels == slabel);
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
				M_ecoc(j, :) = ECOCs(i_ecoc, :);
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
% reservoir sampling, update step
function [samplegist, samplelabel, priority_queue] = update_reservoir(...
		samplegist, samplelabel, priority_queue, spoint, slabel, i, reservoir_size)
	% reservoir update (based on random sort)
	if i <= reservoir_size
		samplegist(i, :)  = spoint;
		samplelabel(i)    = slabel;
		priority_queue(i) = rand;
	else
		% pop max from priority queue
		[maxval, maxind] = max(priority_queue);
		r = rand;
		if maxval > r
			% push into priority queue
			priority_queue(maxind) = r;
			samplegist(maxind, :)  = spoint;
			samplelabel(maxind)    = slabel;
		end
	end
end
