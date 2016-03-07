function train_osh(Xtrain, Ytrain, opts)
	% online (semi-)supervised hashing
	
	train_time  = zeros(1, opts.ntrials);
	update_time = zeros(1, opts.ntrials);
	bit_flips   = zeros(1, opts.ntrials);
	parfor t = 1:opts.ntrials
		myLogInfo('%s: %d trainPts, random trial %d', opts.identifier, opts.noTrainingPoints, t);

		if opts.ntests > 2
			% randomly set test iteration numbers (to better mimic real scenarios)
			% tests at 1, end, and around the endpoints of every interval
			test_iters = [];
			interval = round(opts.noTrainingPoints/(opts.ntests-1));
			for i = 1:opts.ntests-2
				iter = interval*i + randi([1 round(interval/3)]) - round(interval/6);
				test_iters = [test_iters, iter];
			end
			test_iters = [1, test_iters, opts.noTrainingPoints];
		else
			% special case ntests<=2: only test on first & last iteration
			test_iters = [1, opts.noTrainingPoints];
		end

		% do SGD optimization
		[train_time(t), update_time(t), bit_flips(t)] = sgd_optim(...
			Xtrain, Ytrain, test_iters, opts, t);
	end
	myLogInfo('Training time (total): %.2f +/- %.2f', mean(train_time), std(train_time));
	if strcmp(opts.mapping, 'smooth')
		myLogInfo('      Bit flips (per): %.4g +/- %.4g', mean(bit_flips), std(bit_flips));
	end
end

% -------------------------------------------------------------
function [train_time, update_time, bitflips] = sgd_optim(...
		Xtrain, Ytrain, test_iters, opts, trialNo)
	% optimization via SGD
	
	prefix = sprintf('%s/trial%d', opts.expdir, trialNo);
	noexist = 0;
	for i = test_iters
		if ~exist(sprintf('%s_iter%d.mat', prefix, i), 'file')
			noexist = noexist + 1;
		end
	end
	if noexist == 0 && exist([prefix '.mat'], 'file') && ...
            opts.override == 0
        
		myLogInfo('Trial %d already done.', trialNo); 
		load([prefix '.mat']); return;
	end

	% init
	[W, H, ECOCs] = init_osh(Xtrain, opts);
	ntrain_all    = size(Xtrain, 1);
	bitflips      = 0;   bitflips_res = 0;
	train_time    = 0;   update_time  = 0;
    maxLabelSize  = 205; % Sun
    
    persistent table_thr;
    table_thr = arrayfun(@bit_fp_thr,opt.bits*ones(1,maxLabelSize),1:maxLabelSize);
    
    % deal with regularizers
	if opts.reg_rs > 0
		% use reservoir sampling regularizer
		reservoir_size = opts.samplesize;
		Xsample        = zeros(reservoir_size, size(Xtrain, 2));
		Ysample        = zeros(reservoir_size, 1);
		priority_queue = zeros(1, reservoir_size);
		Hres           = [];  % mapped binary codes for the reservoir
	end
	if opts.reg_maxent > 0
		% use max entropy regularizer
		num_unlabeled = 0;
		U = zeros(size(Xtrain, 2));
	end

	% SGD iterations
	i_ecoc = 1;  M_ecoc = [];  seenLabels = [];
	for i = 1:opts.noTrainingPoints
		t_ = tic;  
		% new training point
		spoint = Xtrain(i, :);
		slabel = Ytrain(i, :);

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
			elseif opts.reg_smooth > 0 && opts.reg_rs
				if i > reservoir_size
                    resY = 2*single(W'*samplegist' > 0)-1;
                    qY = 2* single(W'*spoint > 0)-1;
                    [~, ind] = sort(resY' * qY,'descend');
                    
                end
			end
		end

		update_table = false;
		if opts.reg_rs > 0  % reservoir update
			[Xsample, Ysample, priority_queue] = update_reservoir(...
				Xsample, Ysample, priority_queue, spoint, slabel, i, reservoir_size);
            
            % a hack -we always use smooth mapping for reservoir samples 
            ropts = opts;
            ropts.mapping = 'smooth';
			Hnew = build_hash_table(W, Xsample, Ysample, seenLabels, M_ecoc, ropts)';
			if isempty(Hres)
				Hres = Hnew;  
                if strcmp(opts.mapping,'smooth'), update_table = true; end
			else
				bitdiff = xor(Hres, Hnew); %(Hres ~= Hnew);
				bf_temp = sum(bitdiff(:))/reservoir_size;
				% signal update when:
				% 1) using update_interval (for rs_baseline)
				% 2) #bitflips > thresh (for rs)
				% NOTE: get_opts() already ensures only one scenario will happen
                
                
				if mod(i,opts.update_interval) == 0 || (opts.flip_thresh > 0 && ...
                        bf_temp > table_thr(length(seenLabels)))
					bitflips_res = bitflips_res + bf_temp;
					update_table = true;
					Hres = Hnew;
				end
			end
		end

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
			W = sgd_update(W, Xsample, Hres, stepsizes, opts.SGDBoost);
		end

		% SGD-3. update W wrt. unsupervised regularizer (if specified)
		% either max entropy or smoothness, but not both
		if opts.reg_maxent > 0  &&  num_unlabeled > 10
			W = W - opts.reg_maxent * U * W;
		elseif opts.reg_smooth > 0 && i > reservoir_size
			W = reg_smooth(W,[spoint;samplegist(ind(1:opts.rs_sm_neigh_size),:)],opts.reg_smooth);
		end
		train_time = train_time + toc(t_);

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		% hash index update
		if i == 1 || i == opts.noTrainingPoints 
			update_table = true;  % update on first & last iteration no matter what
		elseif opts.reg_rs <= 0
			% NOTE: if using reservoir, update_table is already set.
            % TODO if mapping is not smooth, set update_interval to
            % noTrainingPoints
			update_table = ~mod(i, opts.update_interval);
		end
		%if strcmp(opts.mapping, 'smooth') && update_table
        if update_table
			t_ = tic;
			Hnew = build_hash_table(W, Xtrain, Ytrain, seenLabels, M_ecoc, opts);
			if ~isempty(H)
				bitdiff = xor(H, Hnew);
				bitflips = bitflips + sum(bitdiff(:))/ntrain_all;
			end
			H = Hnew;
			update_time = update_time + toc(t_);
		end

		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		% cache intermediate model to disk
		if ismember(i, test_iters)
			savefile = sprintf('%s_iter%d.mat', prefix, i);
			save(savefile, 'W', 'H', 'bitflips', 'train_time', 'update_time');
			unix(['chmod o-w ' savefile]);  % matlab permission bug
		end
		if ~mod(i, round(opts.noTrainingPoints/5))
			myLogInfo('Trial %02d, Iter %d/%d. Elapsed: SGD %.2f, HT_update %.2f', ...
				trialNo, i, opts.noTrainingPoints, train_time, update_time);
		end
	end % end for

	if opts.reg_rs > 0
		myLogInfo('Trial %02d. bitflips_res = %g', trialNo, bitflips_res);
	end
	if opts.reg_maxent > 0
		myLogInfo('Trial %02d. %d labeled, %d unlabeled. reg_maxent = %g', ...
			trialNo, opts.noTrainingPoints-num_unlabeled, num_unlabeled, opts.reg_maxent);
	end

	%{ KH: obsolete now because we now always update on last iteration
	% populate hash table
	t_ = tic;
	H = build_hash_table(W, Xtrain, Ytrain, seenLabels, M_ecoc, opts);
	update_time = update_time + toc(t_);
	%}
	myLogInfo('Trial %02d. TOTAL: SGD %.2f sec, HT_update %.2f sec', ...
		trialNo, train_time, update_time);

	% save final model, etc
	save([prefix '.mat'], 'W', 'H', 'bitflips', 'train_time', 'update_time', 'test_iters');
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
function [W, H, ECOCs] = init_osh(Xtrain, opts, bigM)
	% randomly generate candidate codewords, store in ECOCs
	if nargin < 3, bigM = 10000; end
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
			M_ecoc = logical(zeros(numel(slabel), size(ECOCs, 2)));
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
% reservoir sampling, update step, based on random sort
function [Xsample, Ysample, priority_queue] = update_reservoir(...
		Xsample, Ysample, priority_queue, spoint, slabel, i, reservoir_size)
	if i <= reservoir_size
		Xsample(i, :)     = spoint;
		Ysample(i)        = slabel;
		priority_queue(i) = rand;
	else
		% pop max from priority queue
		[maxval, maxind] = max(priority_queue);
		r = rand;
		if maxval > r
			% push into priority queue
			priority_queue(maxind) = r;
			Xsample(maxind, :)     = spoint;
			Ysample(maxind)        = slabel;
		end
	end
end

% -----------------------------------------------------------
% smoothness regularizer
function W = reg_smooth(W, points, reg_smooth)
    reg_smooth = reg_smooth/size(points,1);

    for i = 1:size(W,2)
        for j = 1:size(points,2)-1
            W(:,i) = W(:,i) + reg_smooth*(points(1,:)*(W(:,i)'*points(j+1,:)) + ...
                (W(:,i)'*points(1,:))*points(j+1,:));
        end
    end
end


