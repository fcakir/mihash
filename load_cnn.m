function [Xtrain, Ytrain, Xtest, Ytest, cateTrainTest] = load_cnn(dataset, opts)

	tic;
	if strcmp(dataset, 'cifar')
		% trainCNN
		load('/research/codebooks/hashing_project/data/cifar-10/descriptors/trainCNN.mat');
		% traininglabels
		load('/research/codebooks/hashing_project/data/cifar-10/descriptors/traininglabelsCNN.mat');
		% testCNN
		load('/research/codebooks/hashing_project/data/cifar-10/descriptors/testCNN.mat');
		% testlabels
		load('/research/codebooks/hashing_project/data/cifar-10/descriptors/testlabelsCNN.mat');
		cnn       = [trainCNN; testCNN];
		cnnlabels = [traininglabels; testlabels];
		tstperclass = 100;
		[Xtrain, Ytrain, Xtest, Ytest, cateTrainTest] = ...
			split_train_test(cnn, cnnlabels, tstperclass);

	elseif strcmp(dataset, 'sun')
		% TODO
		[Xtrain, Ytrain, Xtest, Ytest, cateTrainTest] = ...
			split_train_test(cnn, cnnlabels, tstperclass);

	elseif strcmp(dataset, 'nus')
		% TODO
		[Xtrain, Ytrain, Xtest, Ytest, cateTrainTest] = ...
			split_train_test_nus(cnn, tags, tstperclass);

	elseif strcmp(dataset, 'places2')
		% TODO

	else, error(['unknown dataset: ' dataset]); end

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
