function osh_gist(dataset, nbits, mapping)
	% PARAMS
	%  dataset (string) from {'cifar' ,'sun','nus'}
	%  mapping (string) from {'bucket','smooth','bucket2'}
	%  nbits (integer) is length of binary code
	%  noTrainingPoints (integer) is the size of training set (2K for cifar, 3970 for sun, trn for nus)
	%  stepsize (float) is step size in SGD
	%  SGDBoost (integer) is 0 for OSHEG, 1 for OSH
	%  
	if nargin < 1, dataset = 'cifar'; end
	if nargin < 2, nbits = 8; end
	if nargin < 3, mapping = 'coord'; end
	if strcmp(dataset, 'cifar')
		load('/research/codebooks/hashing_project/data/cifar-10/descriptors/gist.mat');
		gist             = [traingist; testgist];
		gistlabels       = [trainlabels; testlabels];
		tstperclass      = 100;
		noTrainingPoints = 2000;
	elseif strcmp(dataset, 'sun')
		load('/research/codebooks/hashing_project/data/sun397/SUN_gist.mat');
		gistlabels       = labels;
		tstperclass      = 10;
		noTrainingPoints = 3970;
	else
		error('unknown dataset');
	end
	stepsize = 0.1;
	SGDBoost = 0;

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

	% randomize again (KH: why?)
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
	ntrials = 50;
	mAP = zeros(1, ntrials);
	train_time = zeros(1, ntrials);
	parfor t = 1:ntrials
		fprintf('%s-%dbit-%s: random trial %d\n', dataset, nbits, mapping, t);
		% train
		t0 = tic;
		[W, Y] = train_osh(traingist, trainlabels, noTrainingPoints, ...
			nbits, mapping, stepsize, SGDBoost);
		train_time(t) = toc(t0);

		% test
		tY = 2*single(W'*testgist' > 0)-1;
		mAP(t) = get_mAP(cateTrainTest, Y, tY);
	end
	fprintf('\nTraining time: %.2f +/- %.2f\n', mean(train_time), std(train_time));
	fprintf('     Test mAP: %.3g +/- %.3g\n', mean(mAP), std(mAP));
end

