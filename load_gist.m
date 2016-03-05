function [Xtrain, Ytrain, Xtest, Ytest, cateTrainTest] = load_gist(dataset, opts)

	tic;
	if strcmp(dataset, 'cifar')
		load('/research/codebooks/hashing_project/data/cifar-10/descriptors/gist.mat');
		gist        = [traingist; testgist];
		gistlabels  = [trainlabels; testlabels];
		tstperclass = 100;
		%opts.noTrainingPoints = 2000;  % # points used for training
		[Xtrain, Ytrain, Xtest, Ytest, cateTrainTest] = ...
			split_train_test(gist, gistlabels, tstperclass);

	elseif strcmp(dataset, 'sun')
		load('/research/codebooks/hashing_project/data/sun397/SUN_gist.mat');
		gistlabels  = labels;
		tstperclass = 10;
		%opts.noTrainingPoints = 3970;  % # points used for training
		[Xtrain, Ytrain, Xtest, Ytest, cateTrainTest] = ...
			split_train_test(gist, gistlabels, tstperclass);

	elseif strcmp(dataset, 'nus')
    gist = load('/research/codebooks/hashing_project/data/nuswide/BoW_int.dat');
    tags = load('/research/codebooks/hashing_project/data/nuswide/AllLabels81.txt');
    tstperclass = 30;
		%opts.noTrainingPoints = 20*81;
		[Xtrain, Ytrain, Xtest, Ytest, cateTrainTest] = ...
			split_train_test_nus(gist, tags, tstperclass);

	else, error('unknown dataset'); end

	whos Xtrain Ytrain Xtest Ytest cateTrainTest
	myLogInfo('Dataset "%s" loaded in %.2f secs', dataset, toc);
end

% --------------------------------------------------------
function [Xtrain, Ytrain, Xtest, Ytest, cateTrainTest] = ...
		split_train_test(gist, gistlabels, tstperclass)

	% normalize features
	gist = bsxfun(@minus, gist, mean(gist,1));  % first center at 0
	gist = normalize(gist);  % then scale to unit length

	% construct test and training set
	clslbl      = unique(gistlabels);
	num_classes = length(clslbl);
	testsize    = num_classes * tstperclass;
	trainsize   = size(gist, 1) - testsize;
	gistdim     = size(gist, 2);
	Xtest       = zeros(testsize, gistdim);
	Ytest       = zeros(testsize, 1);
	Xtrain      = zeros(trainsize, gistdim);
	Ytrain      = zeros(trainsize, 1);
	count = 0;
	for i = 1:num_classes
		ind = find(gistlabels == clslbl(i));
		n_i = length(ind);
		ind = ind(randperm(n_i));
		Xtest((i-1)*tstperclass+1:i*tstperclass,:) = gist(ind(1:tstperclass),:);
		Ytest((i-1)*tstperclass+1:i*tstperclass)   = clslbl(i);
		Xtrain(count+1:count+n_i-tstperclass, :)   = gist(ind(tstperclass+1:end),:);
		Ytrain(count+1:count+n_i-tstperclass, :)   = clslbl(i);
		count = count + n_i - tstperclass;
	end
	% randomize again
	ind    = randperm(size(Xtrain, 1));
	Xtrain = Xtrain(ind, :);
	Ytrain = Ytrain(ind);
	ind    = randperm(size(Xtest, 1));
	Xtest  = Xtest(ind, :);
	Ytest  = Ytest(ind);

	cateTrainTest = repmat(Ytrain, 1, length(Ytest)) ...
		== repmat(Ytest, 1, length(Ytrain))';
end

% --------------------------------------------------------
function [Xtrain, Ytrain, Xtest, Ytest, cateTrainTest] = ...
		split_train_test_nus(gist, tags, tstperclass)

	% normalize features
	gist = bsxfun(@minus, gist, mean(gist,1));  % first center at 0
	gist = normalize(gist);  % then scale to unit length

	% construct test and training set
	num_classes = 81;
	testsize    = num_classes * tstperclass;
	ind         = randperm(size(gist, 1));
	Xtest       = gist(ind(1:testsize), :);
	Ytest       = tags(ind(1:testsize), :);
	Xtrain      = gist(ind(testsize+1:end), :);
	Ytrain      = tags(ind(testsize+1:end), :);

	% randomize again
	ind    = randperm(size(Xtrain, 1));
	Xtrain = Xtrain(ind, :);
	Ytrain = Ytrain(ind, :);
	ind    = randperm(size(Xtest, 1));
	Xtest  = Xtest(ind, :);
	Ytest  = Ytest(ind, :);

	cateTrainTest = (Ytrain * Ytest' > 0);
end
