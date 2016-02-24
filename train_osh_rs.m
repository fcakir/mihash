function [expdir] = train_osh_rs(traingist, trainlabels, opts)
	% online supervised hashing
	% regularization term defined on reservoir samples
	expdir = sprintf('%s/%s-u%d-RS%g', opts.localdir, opts.identifier, ...
		opts.update_interval, opts.sampleratio);
	if ~exist(expdir, 'dir'), mkdir(expdir); unix(['chmod g+rw ' expdir]); end

	train_time  = zeros(1, opts.ntrials);
	update_time = zeros(1, opts.ntrials);
	bit_flips   = zeros(1, opts.ntrials);
	parfor t = 1:opts.ntrials
		myLogInfo('%s: random trial %d', opts.identifier, t);
		[train_time(t), update_time(t), bit_flips(t)] = train_sgd_rs(...
			traingist, trainlabels, opts, expdir, t);
	end
	myLogInfo('Training time (total): %.2f +/- %.2f', mean(train_time), std(train_time));
	if strcmp(opts.mapping, 'smooth')
		myLogInfo('    Bit flips (total): %.4g +/- %.4g', mean(bit_flips), std(bit_flips));
	end
end

% -------------------------------------------------------------
function [train_time, update_time, bitflips] = train_sgd_rs(traingist, trainlabels, opts, expdir, trialNo)
	prefix = sprintf('%s/trial%d', expdir, trialNo);
	if exist([prefix '.mat'], 'file')
		myLogInfo('Trial %d already done.', trialNo); 
		load([prefix '.mat']);
		return;
	end

	% randomly generate candidate codewords, store in M2
	bigM = 10000;
	M2   = zeros(bigM, opts.nbits);
	for t = 1:opts.nbits
		r = ones(bigM, 1);
		while (abs(sum(r)) == bigM)
			r = 2*randi([0,1], bigM, 1)-1;
		end
		M2(:, t) = r;
	end
	clear r

	% initialize with LSH
	d = size(traingist, 2);
	W = randn(d, opts.nbits);
	W = W ./ repmat(diag(sqrt(W'*W))',d,1);
	Y = [];  % the indexing structure

	bitflips = 0;
	train_time = 0;
	update_time = 0;

	% do simple sampling
	% KH: TODO reservoir sampling
	bitflips = 0;
	ntrain_all = size(traingist, 1);
	sid = randperm(ntrain_all, ceil(opts.sampleratio*ntrain_all));
	samplegist = traingist(sid, :);

	i_ecoc = 1;
	classLabels = [];
	for i = 1:opts.noTrainingPoints
		t_ = tic;
		% new training point
		spoint = traingist(i, :);
		slabel = trainlabels(i);

		% check whether it exists in the "seen class labels" vector
		islabel = find(classLabels == slabel);
		if isempty(islabel)
			if isempty(classLabels)
				% does not exist, create a binary code for M
				classLabels = slabel;
				M = M2(i_ecoc, :);
				i_ecoc = i_ecoc + 1;
			else
				% append codeword to ECOC matrix
				classLabels = [classLabels; slabel];
				M = [M; M2(i_ecoc,:)];
				i_ecoc = i_ecoc +1;
			end
		end
		islabel = find(classLabels == slabel);

		% hash function update
		if opts.SGDBoost == 0
			for j = 1:opts.nbits
				if M(islabel,j)*W(:,j)'*spoint' > 1
					continue;
				else
					W(:,j) = W(:,j) + opts.stepsize * M(islabel,j)*spoint';
				end
				%W = W ./ repmat(diag(sqrt(W'*W))',d,1);
			end
		else
			for j = 1:opts.nbits
				if j ~= 1
					c1 = exp(-(M(islabel,1:j-1)*(W(:,1:j-1)'*spoint')));
				else
					c1 = 1;
				end
				W(:,j) = W(:,j) - opts.stepsize * ...
					c1 * exp(-M(islabel,j)*W(:,j)'*spoint')*-M(islabel,j)*spoint';
				%W = W ./ repmat(diag(sqrt(W'*W))',d,1);
			end
		end
		train_time = train_time + toc(t_);

		% hash index update
		if strcmp(opts.mapping, 'smooth') && ~mod(i, opts.update_interval)
			t_ = tic;
			if isempty(Y)
				Y = 2*single(W'*samplegist' > 0)-1;
			else
				Ynew = 2*single(W'*samplegist' > 0)-1;
				bitdiff = (Y ~= Ynew);
				bitflips = bitflips + sum(bitdiff(:));
				Y = Ynew;
			end
			update_time = update_time + toc(t_);
		end

		% cache intermediate model to disk
		if ~mod(i, opts.test_interval)
			if isempty(Y)
				Y = 2*single(W'*samplegist' > 0)-1;
			end
			save(savefile, 'W', 'Y');
		end
	end % end for


	% populate hash table
	if strcmp(opts.mapping,'smooth')
		myLogInfo('%d bitflips, ', bitflips);
		Y = 2*single(W'*traingist' > 0)-1;

	elseif strcmp(opts.mapping,'bucket')
		Y = zeros(nbits, size(traingist,1), 'single');
		for i = 1:length(classLabels)
			ind = find(classLabels(i) == trainlabels);
			Y(:,ind) = repmat(M(i,:)',1,length(ind));
		end

	elseif strcmp(opts.mapping,'bucket2')
		Y = 2*single(W'*traingist' > 0)-1;
		sim = M * Y;
		Y = zeros(nbits, size(traingist,1), 'single');
		[~, maxInd] = max(sim);
		Y = M(maxInd,:)';

	elseif strcmp(opts.mapping, 'coord') 
		% KH: do extra coordinate descent step on codewords
		Y = 2*single(W'*traingist' > 0)-1;
		for i = 1:length(classLabels)
			ind = find(classLabels(i) == trainlabels);
			% find codeword that minimizes J
			cw = 2*single(mean(Y(:, ind), 2) > 0)-1; 
			Y(:,ind) = repmat(cw, 1,length(ind));
		end
	end
end
