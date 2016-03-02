function [traingist, trainlabels, testgist, testlabels, cateTrainTest, opts] = ...
		load_gist(dataset, opts)

	tic;
	if strcmp(dataset, 'cifar')
		load('/research/codebooks/hashing_project/data/cifar-10/descriptors/gist.mat');
		gist        = [traingist; testgist];
		gistlabels  = [trainlabels; testlabels];
		tstperclass = 100;
		opts.noTrainingPoints = 2000;  % # points used for training
		[traingist, trainlabels, testgist, testlabels, cateTrainTest] = ...
			split_train_test(gist, gistlabels, tstperclass);

	elseif strcmp(dataset, 'sun')
		load('/research/codebooks/hashing_project/data/sun397/SUN_gist.mat');
		gistlabels  = labels;
		tstperclass = 10;
		opts.noTrainingPoints = 3970;  % # points used for training
		[traingist, trainlabels, testgist, testlabels, cateTrainTest] = ...
			split_train_test(gist, gistlabels, tstperclass);

	elseif strcmp(dataset, 'nus')
    gist = load('/research/codebooks/hashing_project/data/nuswide/BoW_int.dat');
    tags = load('/research/codebooks/hashing_project/data/nuswide/AllLabels81.txt');
    tstperclass = 30;
		opts.noTrainingPoints = 20*81;
		[traingist, trainlabels, testgist, testlabels, cateTrainTest] = ...
			split_train_test_nus(gist, tags, tstperclass);

	else, error('unknown dataset'); end

	whos traingist trainlabels testgist testlabels cateTrainTest
	myLogInfo('Dataset "%s" loaded in %.2f secs', dataset, toc);
end

% --------------------------------------------------------
function [traingist, trainlabels, testgist, testlabels, cateTrainTest] = ...
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
	ind         = randperm(size(traingist, 1));
	traingist   = traingist(ind, :);
	trainlabels = trainlabels(ind);
	ind         = randperm(size(testgist, 1));
	testgist    = testgist(ind, :);
	testlabels  = testlabels(ind);

	cateTrainTest = repmat(trainlabels, 1, length(testlabels)) ...
		== repmat(testlabels, 1, length(trainlabels))';
end

% --------------------------------------------------------
function [traingist, traintags, testgist, testtags, cateTrainTest] = ...
		split_train_test_nus(gist, tags, tstperclass)

	% normalize features
	gist = bsxfun(@minus, gist, mean(gist,1));  % first center at 0
	gist = normalize(gist);  % then scale to unit length

	% construct test and training set
	num_classes = 81;
	testsize    = num_classes * tstperclass;
	ind         = randperm(size(gist, 1));
	testgist    = gist(ind(1:testsize), :);
	testtags    = tags(ind(1:testsize), :);
	traingist   = gist(ind(testsize+1:end), :);
	traintags   = tags(ind(testsize+1:end), :);

	% randomize again
	ind       = randperm(size(traingist, 1));
	traingist = traingist(ind, :);
	traintags = traintags(ind, :);
	ind       = randperm(size(testgist, 1));
	testgist  = testgist(ind, :);
	testtags  = testtags(ind, :);

	cateTrainTest = (traintags * testtags' > 0);
end
