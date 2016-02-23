function osh_gist(dataset, nbits, varargin)
	% PARAMS
	%  dataset (string) from {'cifar' ,'sun','nus'}
	%  nbits (integer) is length of binary code
	%  varargin: see get_opts.m for details
	%  
	if nargin < 1, dataset = 'cifar'; end
	if nargin < 2, nbits = 8; end
	opts = get_opts(dataset, nbits, varargin{:});  % set parameters

	if strcmp(dataset, 'cifar')
		load('/research/codebooks/hashing_project/data/cifar-10/descriptors/gist.mat');
		gist        = [traingist; testgist];
		gistlabels  = [trainlabels; testlabels];
		tstperclass = 100;
		opts.noTrainingPoints = 2000;  % # points used for training
	elseif strcmp(dataset, 'sun')
		load('/research/codebooks/hashing_project/data/sun397/SUN_gist.mat');
		gistlabels  = labels;
		tstperclass = 10;
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

	%% ONLINE LEARNING
	expdir = train_osh(traingist, trainlabels, opts);  % baseline
	%train_osh_rs(traingist, trainlabels, opts);  % with reservoir regularizer
	%train_osh_l1l2(traingist, trainlabels, opts);  % with L1L2 regularizer

	%% test models
	mAPfn = sprintf('%s/mAP_t%d.mat', expdir, opts.test_interval);
	try 
		load(mAPfn);
	catch
		n = floor(opts.noTrainingPoints/opts.test_interval);
		mAP = zeros(opts.ntrials, n);
		for t = 1:opts.ntrials
			for i = 1:n
				F = sprintf('%s/trial%d_iter%d.mat', expdir, t, i*opts.test_interval);
				d = load(F);
				W = d.W;
				Y = d.Y;
				tY = 2*single(W'*testgist' > 0)-1;
		
				% NOTE: get_mAP() uses parfor
				mAP(t, i) = get_mAP(cateTrainTest, Y, tY);
			end
		end
		save(mAPfn, 'mAP');
	end
	myLogInfo('Test mAP (final): %.3g +/- %.3g', mean(mAP(:,end)), std(mAP(:,end)));
end
