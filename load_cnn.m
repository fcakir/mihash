function [traincnn, trainlabels, testcnn, testlabels, cateTrainTest, opts] = ...
		load_cnn(dataset, opts)

	tic;
	if strcmp(dataset, 'cifar')
		load('/research/codebooks/hashing_project/data/cifar-10/descriptors/cnn.mat');
		cnn        = [traincnn; testcnn];
		cnnlabels  = [trainlabels; testlabels];
		tstperclass = 100;
		%opts.noTrainingPoints = 2000;  % # points used for training
		[traincnn, trainlabels, testcnn, testlabels, cateTrainTest] = ...
			split_train_test(cnn, cnnlabels, tstperclass);

	elseif strcmp(dataset, 'sun')
		load('/research/codebooks/hashing_project/data/sun397/SUN_cnn.mat');
		cnnlabels  = labels;
		tstperclass = 10;
		%opts.noTrainingPoints = 3970;  % # points used for training
		[traincnn, trainlabels, testcnn, testlabels, cateTrainTest] = ...
			split_train_test(cnn, cnnlabels, tstperclass);

	elseif strcmp(dataset, 'nus')
    cnn = load('/research/codebooks/hashing_project/data/nuswide/BoW_int.dat');
    tags = load('/research/codebooks/hashing_project/data/nuswide/AllLabels81.txt');
    tstperclass = 30;
		%opts.noTrainingPoints = 20*81;
		[traincnn, trainlabels, testcnn, testlabels, cateTrainTest] = ...
			split_train_test_nus(cnn, tags, tstperclass);

	elseif strcmp(dataset, 'places2')
		% TODO

	else, error('unknown dataset'); end

	whos traincnn trainlabels testcnn testlabels cateTrainTest
	myLogInfo('Dataset "%s" loaded in %.2f secs', dataset, toc);
end

% --------------------------------------------------------
function [traincnn, trainlabels, testcnn, testlabels, cateTrainTest] = ...
		split_train_test(cnn, cnnlabels, tstperclass)

	% normalize features
	cnn = bsxfun(@minus, cnn, mean(cnn,1));  % first center at 0
	cnn = normalize(cnn);  % then scale to unit length

	% construct test and training set
	clslbl      = unique(cnnlabels);
	num_classes = length(clslbl);
	testsize    = num_classes * tstperclass;
	trainsize   = size(cnn, 1) - testsize;
	cnndim     = size(cnn, 2);
	testcnn    = zeros(testsize, cnndim);
	testlabels  = zeros(testsize, 1);
	traincnn   = zeros(trainsize, cnndim);
	trainlabels = zeros(trainsize, 1);
	count = 0;
	for i = 1:num_classes
		ind = find(cnnlabels == clslbl(i));
		n_i = length(ind);
		ind = ind(randperm(n_i));
		testcnn((i-1)*tstperclass+1:i*tstperclass,:) = cnn(ind(1:tstperclass),:);
		testlabels((i-1)*tstperclass+1:i*tstperclass) = clslbl(i);
		traincnn(count+1:count+n_i-tstperclass, :)   = cnn(ind(tstperclass+1:end),:);
		trainlabels(count+1:count+n_i-tstperclass, :) = clslbl(i);
		count = count + n_i - tstperclass;
	end
	% randomize again
	ind         = randperm(size(traincnn, 1));
	traincnn   = traincnn(ind, :);
	trainlabels = trainlabels(ind);
	ind         = randperm(size(testcnn, 1));
	testcnn    = testcnn(ind, :);
	testlabels  = testlabels(ind);

	cateTrainTest = repmat(trainlabels, 1, length(testlabels)) ...
		== repmat(testlabels, 1, length(trainlabels))';
end

% --------------------------------------------------------
function [traincnn, traintags, testcnn, testtags, cateTrainTest] = ...
		split_train_test_nus(cnn, tags, tstperclass)

	% normalize features
	cnn = bsxfun(@minus, cnn, mean(cnn,1));  % first center at 0
	cnn = normalize(cnn);  % then scale to unit length

	% construct test and training set
	num_classes = 81;
	testsize    = num_classes * tstperclass;
	ind         = randperm(size(cnn, 1));
	testcnn    = cnn(ind(1:testsize), :);
	testtags    = tags(ind(1:testsize), :);
	traincnn   = cnn(ind(testsize+1:end), :);
	traintags   = tags(ind(testsize+1:end), :);

	% randomize again
	ind       = randperm(size(traincnn, 1));
	traincnn = traincnn(ind, :);
	traintags = traintags(ind, :);
	ind       = randperm(size(testcnn, 1));
	testcnn  = testcnn(ind, :);
	testtags  = testtags(ind, :);

	cateTrainTest = (traintags * testtags' > 0);
end
