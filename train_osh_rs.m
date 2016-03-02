function train_osh_rs(traingist, trainlabels, opts)
	% online supervised hashing
	% regularization term defined on reservoir samples
	
	train_time  = zeros(1, opts.ntrials);
	update_time = zeros(1, opts.ntrials);
	bit_flips   = zeros(1, opts.ntrials);
	parfor t = 1:opts.ntrials
		myLogInfo('%s: random trial %d', opts.identifier, t);
		[train_time(t), update_time(t), bit_flips(t)] = train_sgd_rs(...
			traingist, trainlabels, opts, t);
	end
	myLogInfo('Training time (total): %.2f +/- %.2f', mean(train_time), std(train_time));
	if strcmp(opts.mapping, 'smooth')
		myLogInfo('      Bit flips (per): %.4g +/- %.4g', mean(bit_flips), std(bit_flips));
	end
end

% -------------------------------------------------------------
function [train_time, update_time, bitflips] = train_sgd_rs(...
		traingist, trainlabels, opts, trialNo)
	% SGD with reservoir regularizer
	prefix = sprintf('%s/trial%d', opts.expdir, trialNo);
	noexist = 0;
	for i = 1:floor(opts.noTrainingPoints/opts.test_interval)
		if ~exist(sprintf('%s_iter%d.mat', prefix, i), 'file')
			noexist = noexist + 1;
		end
	end
	if noexist == 0 && exist([prefix '.mat'], 'file')
		myLogInfo('Trial %d already done.', trialNo); 
		load([prefix '.mat']);
		return;
	end

	% randomly generate candidate codewords, store in ECOCs
	bigM = 10000;
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

	bitflips = 0;
	bitflips_res = 0;
	train_time = 0;
	update_time = 0;

	% reservoir sampling
	ntrain_all     = size(traingist, 1);
	reservoir_size = opts.samplesize; %ceil(opts.sampleratio*ntrain_all);
	samplegist     = zeros(reservoir_size, size(traingist, 2));
	samplelabel    = zeros(reservoir_size, 1);
	Yres           = [];  % mapped binary codes for the reservoir

	i_ecoc = 1;
	M_ecoc = [];
	seenLabels = [];
	for i = 1:opts.noTrainingPoints
		t_ = tic;
		skip_update = false;
		% new training point
		spoint = traingist(i, :);
		slabel = trainlabels(i, :);

		% check whether it exists in the "seen class labels" vector
		if length(slabel) == 1  % single-label dataset
			islabel = find(seenLabels == slabel);
			if isempty(islabel)
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
			islabel = find(seenLabels == slabel);
		else  % multi-label dataset
			if sum(slabel) == 0, skip_update = true; end
			if isempty(seenLabels), 
				seenLabels = zeros(size(slabel)); 
			end

			% if not all incoming labels are seen
			inds = find(slabel == 1);
			seen = seenLabels(inds);
			if sum(seen) < length(seenLabels)
				seenLabels(inds) = 1;
				for j = find(seen == 0)
					M_ecoc = [M_ecoc; ECOCs(inds(j), :)];
					indexRank(i_ecoc) = inds(j);
					i_ecoc = i_ecoc+1;
				end
			end
			islabel = find(ismember(indexRank, inds));
		end
		target_codes = M_ecoc(islabel, :);

		% reservoir update (based on random sort)
		priority_queue = zeros(1, reservoir_size);
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
			% compute binary codes for the reservoir
			if isempty(Yres)
				Yres = build_hash_table(W, samplegist, samplelabel, seenLabels, M_ecoc, opts)';
			else
				Ynew = build_hash_table(W, samplegist, samplelabel, seenLabels, M_ecoc, opts)';
				bitdiff = (Yres ~= Ynew);
				bitflips_res = bitflips_res + sum(bitdiff(:));
				Yres = Ynew;
			end
		end

		% hash function update
		if ~skip_update
			if opts.SGDBoost == 0
				for c = 1:size(target_codes, 1)
					code = target_codes(c, :);
					% vectorized updates
					if i <= reservoir_size
						W = sgd_update_hinge(W, spoint, code, opts.stepsize);
					else
						codes  = [code; Yres];
						points = [spoint; samplegist*(opts.lambda/reservoir_size)];
						stepsizes = opts.stepsize * [1; ones(reservoir_size,1)*opts.lambda/reservoir_size];
						W = sgd_update_hinge(W, points, codes, stepsizes);
					end
				end
			else
				for c = 1:size(target_codes, 1)
					code = target_codes(c, :);
					% TODO vectorize
					for j = 1:opts.nbits
						if j ~= 1
							c1 = exp(-(code(1:j-1)*(W(:,1:j-1)'*spoint')));
						else
							c1 = 1;
						end
						W(:,j) = W(:,j) - opts.stepsize * ...
							c1 * exp(-code(j)*W(:,j)'*spoint')*-code(j)*spoint';
					end
				end
			end
			train_time = train_time + toc(t_);
		end

		% hash index update
		if strcmp(opts.mapping, 'smooth') && ~mod(i, opts.update_interval)
			t_ = tic;
			if isempty(Y)
				Y = build_hash_table(W, traingist, trainlabels, seenLabels, M_ecoc, opts);
			else
				Ynew = build_hash_table(W, traingist, trainlabels, seenLabels, M_ecoc, opts);
				bitdiff = (Y ~= Ynew);
				bitflips = bitflips + sum(bitdiff(:));
				Y = Ynew;
			end
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
	bitflips_res = bitflips_res/reservoir_size;

	% populate hash table
	t_ = tic;
	Y = build_hash_table(W, traingist, trainlabels, seenLabels, M_ecoc, opts);
	update_time = update_time + toc(t_);
	myLogInfo('Trial %02d. bitflips_res=%.2f, SGD+reservoir: %.2f sec, Hashtable update: %.2f sec', ...
		trialNo, bitflips_res, train_time, update_time);

	% save final model, etc
	save([prefix '.mat'], 'W', 'Y', 'bitflips', 'train_time', 'update_time');
	unix(['chmod o-w ' prefix '.mat']);  % matlab permission bug
end

% -----------------------------------------------------------
function W = sgd_update_hinge(W, points, codes, stepsizes)
	% input: 
	%   W         - D*nbits matrix, each col is a hyperplane
	%   points    - n*D matrix, each row is a point
	%   codes     - n*nbits matrix, each row the corresp. target binary code
	%   stepsizes - SGD step sizes (1 per point) for current batch
	% output: 
	%   updated W
	for i = 1:size(points, 1)
		xi = points(i, :);
		ci = codes(i, :);
		id = (xi * W .* ci <= 1);  % logical indexing > find()
		n  = sum(id);
		if n > 0
			W(:,id) = W(:,id) + stepsizes(i)*repmat(xi',[1 n])*diag(ci(id)); 
		end
	end
end
