function [traingist, trainlabels, testgist, testlabels, opts] = load_gist(dataset, opts)
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
	whos gist traingist testgist cateTrainTest
end
