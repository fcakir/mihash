function osh_gist(dataset, nbits, varargin)
	% PARAMS
	%  dataset (string) from {'cifar' ,'sun','nus'}
	%  nbits (integer) is length of binary code
	%  
	if nargin < 1, dataset = 'cifar'; end
	if nargin < 2, nbits = 8; end
	opts = get_opts(dataset, nbits, varargin);  % set parameters

	if strcmp(dataset, 'cifar')
		load('/research/codebooks/hashing_project/data/cifar-10/descriptors/gist.mat');
		gist       = [traingist; testgist];
		gistlabels = [trainlabels; testlabels];
		opts.tstperclass      = 100;
		opts.noTrainingPoints = 2000;  % # points used for training
	elseif strcmp(dataset, 'sun')
		load('/research/codebooks/hashing_project/data/sun397/SUN_gist.mat');
		gistlabels = labels;
		opts.tstperclass      = 10;
		opts.noTrainingPoints = 3970;  % # points used for training
	else
		error('unknown dataset');
	end

	% normalize features
	gist = bsxfun(@minus, gist, mean(gist,1));  % first center at 0
	gist = normalize(gist);  % then scale to unit length

	% construct test and training set
	clslbl      = unique(gistlabels);
	num_classes = length(clslbl);
	testsize    = num_classes * tstperclass;
	trainsize   = size(gist, 1) - testsize;
	gistdim     = size(gist, 2);
	testgist    = zeros(testsize, gistdim);
	testlabels  = zeros(testsize, 1);
	traingist   = zeros(trainsize, gistdim);
	trainlabels = zeros(trainsize, 1);
	count = 0;
	for i = 1:num_classes
		ind = find(gistlabels == clslbl(i));
		n_i = length(ind);
		ind = ind(randperm(n_i));
		testgist((i-1)*tstperclass+1:i*tstperclass,:) = gist(ind(1:tstperclass),:);
		testlabels((i-1)*tstperclass+1:i*tstperclass) = clslbl(i);
		traingist(count+1:count+n_i-tstperclass, :)   = gist(ind(tstperclass+1:end),:);
		trainlabels(count+1:count+n_i-tstperclass, :) = clslbl(i);
		count = count + n_i - tstperclass;
	end

	% randomize again
	if 1
		ind         = randperm(trainsize);
		traingist   = traingist(ind, :);
		trainlabels = trainlabels(ind);
		ind         = randperm(testsize);
		testgist    = testgist(ind,:);
		testlabels  = testlabels(ind);
	end
	cateTrainTest = repmat(trainlabels, 1, length(testlabels)) ...
		== repmat(testlabels, 1, length(trainlabels))';
	whos gist traingist testgist cateTrainTest

	%%% ONLINE LEARNING %%%
	trial_file = cell(1, ntrials);
	train_time = zeros(1, ntrials);
	bit_flips  = zeros(1, ntrials);
	parfor t = 1:ntrials
		trial_file{t} = sprintf('%s/%s-%dbit-%s-trial%d.mat', ...
			localdir, dataset, nbits, mapping, t);
		if exist(trial_file{t}), continue; end
		myLogInfo('%s-%dbit-%s: random trial %d\n', dataset, nbits, mapping, t);

		% train
		t0 = tic;
		%[W, Y, bit_flips(t)] = train_osh(traingist, trainlabels, noTrainingPoints, ...
		[W, Y, bit_flips(t)] = train_osh_rs(traingist, trainlabels, noTrainingPoints, ...
			opts);
			%nbits, mapping, stepsize, SGDBoost, 0.01);
		train_time(t) = toc(t0);

		% save to scratch dir
		res = [];
		res.W = W;
		res.Y = Y;
		res.bit_flips = bit_flips(t);
		res.train_time = train_time(t);
		parsave(trial_file{t}, res);
	end

	% test models
	% NOTE: get_mAP() uses parfor
	mAP = zeros(1, ntrials);
	for t = 1:ntrials
		d = load(trial_file{t});
		W = d.W;
		Y = d.Y;
		tY = 2*single(W'*testgist' > 0)-1;
		mAP(t) = get_mAP(cateTrainTest, Y, tY);
	end

	myLogInfo('Training time: %.2f +/- %.2f', mean(train_time), std(train_time));
	myLogInfo('     Test mAP: %.3g +/- %.3g', mean(mAP), std(mAP));
	if strcmp(mapping, 'smooth')
		myLogInfo('    Bit flips: %.3g +/- %.3g', mean(bit_flips), std(bit_flips));
	end
end


function parsave(fname, res)
	save(fname, '-struct', 'res');
	unix(['chmod o-w ' fname]);  % fix for matlab permission bug
end

