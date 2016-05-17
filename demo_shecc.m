function resfn = demo_shecc(ftype, dataset, nbits, varargin)

	opts = get_opts_shecc(ftype, dataset, nbits, varargin{:});  % set parameters

	% add libsvm-weights to path
	addpath(genpath('/research/codebooks/hashing_project/code/libsvm-weights/'));


	% 0. result files
	Rprefix = sprintf('%s/%s', opts.expdir, opts.metric);
	if opts.test_frac < 1
		Rprefix = sprintf('%s_frac%g', Rprefix, opts.test_frac);
	end

	resfn = sprintf('%s_%dtrials.mat', Rprefix, opts.ntrials);
	res_trial_fn = cell(1, opts.ntrials);
	for t = 1:opts.ntrials 
		res_trial_fn{t} = sprintf('%s_trial%d.mat', Rprefix, t);
	end

	if opts.override
		res_exist = zeros(1, opts.ntrials);
	else
		res_exist = cellfun(@(r) exist(r, 'file'), res_trial_fn);
	end


	% 1. determine which (training) trials to run
	if opts.override
		run_trial = ones(1, opts.ntrials);
	else
		run_trial = zeros(1, opts.ntrials);
		for t = 1:opts.ntrials
			trial_model_file = sprintf('%s/trial%d.mat', opts.expdir, t);
			if exist(trial_model_file, 'file')
				run_trial(t) = 0;
			else
				run_trial(t) = 1;
			end
		end
	end


	% 2. load data (only if necessary)
	global Xtrain Xtest Ytrain Ytest Dtype
	Dtype_this = [dataset '_' ftype];
	if ~isempty(Dtype) && strcmp(Dtype_this, Dtype)
		myLogInfo('Dataset already loaded for %s', Dtype_this);
	elseif (any(run_trial) || ~all(res_exist))
		myLogInfo('Loading data for %s...', Dtype_this);
		eval(['[Xtrain, Ytrain, Xtest, Ytest] = load_' opts.ftype '(dataset, opts);']);
		Dtype = Dtype_this;

	end
	% 3. TRAINING: run all _necessary_ trials (handled by train_osh)
	if any(run_trial)
		myLogInfo('Training models...');
		train_shecc(run_trial, opts);
	end
	myLogInfo('Training is done.');

	% 4. TESTING: run all _necessary_ trials
	if ~all(res_exist) || ~exist(resfn, 'file')
		% NOTE reusing test_osh for AdaptHash
		myLogInfo('Testing models...');
		test_shecc(resfn, res_trial_fn, res_exist, opts);
	end
	myLogInfo('Testing is done.');
end



function opts = get_opts_shecc(ftype, dataset, nbits, varargin)

	ip = inputParser;

	% default values
	ip.addParamValue('ftype', ftype, @isstr);
	ip.addParamValue('dataset', dataset, @isstr);
	ip.addParamValue('nbits', nbits, @isscalar);

	ip.addParamValue('nworkers', 6, @isscalar);
	ip.addParamValue('randseed', 12345, @isscalar);

	ip.addParamValue('override', 0, @isscalar);
	ip.addParamValue('ntrials', 5, @isscalar);
	ip.addParamValue('noTrainingPoints', 2000, @isscalar);
	ip.addParamValue('mapping', 'smooth', @isstr);

	ip.addParamValue('metric', 'mAP', @isstr);    % evaluation metric
	ip.addParamValue('test_frac', 1, @isscalar);  % <1 for faster testing
	ip.addParamValue('learner','tree',@isstr);
	ip.addParamValue('localdir', ...
		'/research/object_detection/cachedir/online-hashing/shecc', @isstr);
	ip.addParamValue('alpha', 0.9, @isscalar);

	% parse input
	ip.parse(varargin{:});
	opts = ip.Results;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% assertions
	assert(ismember(opts.ftype, {'gist', 'cnn'}));
	assert(ismember(opts.mapping,{'smooth','bucket'}));
	assert(opts.test_frac > 0);
	assert(opts.nworkers>0 && opts.nworkers<=12);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	% matlabpool handling
	if matlabpool('size') == 0
		myLogInfo('Opening matlabpool, nworkers = %d', opts.nworkers);
		matlabpool close force local  % clear up zombies
		matlabpool(opts.nworkers);
	end

	% are we on window$?
	opts.windows = ~isempty(strfind(computer, 'WIN'));
	if opts.windows
		% reset localdir
		opts.localdir = '\\kraken\object_detection\cachedir\online-hashing';
		myLogInfo('We are on Window$. localdir set to %s', opts.localdir);
	end
	
	% make localdir
	if ~exist(opts.localdir, 'dir'), 
		mkdir(opts.localdir);  
		if ~opts.windows, unix(['chmod g+rw ' opts.localdir]); end
	end

	% set randseed -- don't change the randseed if don't have to!
	rng(opts.randseed);

	% identifier string for the current experiment
	opts.identifier = sprintf('%s-%s-%d%s-A%g-L%s', opts.dataset, opts.ftype, ...
		opts.nbits, opts.mapping, opts.alpha,opts.learner);
	
	opts.identifier = sprintf('%s-%dpts', opts.identifier, ...
		opts.noTrainingPoints);
	myLogInfo('identifier: %s', opts.identifier);

	% set expdir
	opts.expdir = sprintf('%s/%s', opts.localdir, opts.identifier);
	if ~exist(opts.expdir, 'dir'), 
		myLogInfo(['creating opts.expdir: ' opts.expdir]);
		mkdir(opts.expdir); 
		if ~opts.windows, unix(['chmod g+rw ' opts.expdir]); end
	end

	% decipher evaluation metric
	if ~isempty(strfind(opts.metric, 'prec_k'))
		% eg. prec_k3 is precision at k=3
		opts.prec_k = sscanf(opts.metric(7:end), '%d');
	elseif ~isempty(strfind(opts.metric, 'prec_n'))
		% eg. prec_n3 is precision at n=3
		opts.prec_n = sscanf(opts.metric(7:end), '%d');
	else 
		assert(strcmp(opts.metric, 'mAP'), 'unknown opts.metric');
	end

end

function train_shecc(run_trial, opts)

	global Xtrain Ytrain
	train_time  = zeros(1, opts.ntrials);
	for t = 1:opts.ntrials
		if run_trial(t) == 0
			myLogInfo('Trial %02d not required, skipped', t);
			continue;
		end
		myLogInfo('%s: %d trainPts, random trial %d', opts.identifier, opts.noTrainingPoints, t);

		prefix = sprintf('%s/trial%d', opts.expdir, t);

		% Learn hash mapping
		[train_time(t)] = shecc(Xtrain, Ytrain, ...
			prefix, t, opts);
	end

	myLogInfo('Training time (total): %.2f +/- %.2f', mean(train_time), std(train_time));
end


% ---------------------------------------------------------
function [traintimes] = shecc(Xtrain, Ytrain, prefix, trialNo, opts)
	% Xtrain (float) n x d matrix where n is number of points 
	%                   and d is the dimensionality 
	%
	% Ytrain (int) is n x 1 matrix containing labels 
	%


	% Use a subset of the training data to learn hash mapping
	% train_data 	: contains subset of training data
	% train_labels 	: contains corresponding label data 
	[n,d]       	= size(Xtrain);
	ind		= randperm(n);
	train_data 	= Xtrain(ind(1:opts.noTrainingPoints),:);
	train_labels 	= floor(Ytrain(ind(1:opts.noTrainingPoints))/10); % ShECC cannot be performed on multi-labelled datasetes
	
	myLogInfo('[T%02d] %d training size, learner ''%s''', trialNo, opts.noTrainingPoints,opts.learner);

	[tn,td] = size(train_data);

	assert(tn == size(train_labels,1));
	
	classLabels = unique(train_labels);
	noOfClasses = length(classLabels);
	assert(noOfClasses == length(unique(Ytrain)), ...
	sprintf('Not all labels are observed in train_data\nTry increasing size\nQuiting...'));
	% M is ECOC matrix 
	M = zeros(noOfClasses, opts.nbits);

	% D is the 2D probability distribution 
	D = zeros(tn, noOfClasses);
	
	tic;	
	D(:) = 1/(tn*(noOfClasses - 1));
	for i = 1:tn
		ind = find(classLabels == train_labels(i));
		D(i,ind) = 0;	    
	end

	for t = 1:opts.nbits
	    
	    % compute coloring
		r = randi([0,1],noOfClasses,1);
		ind = find(r == 0);
		r(ind) = -1;

		while (abs(sum(r)) == noOfClasses)
			r = randi([0,1],noOfClasses,1);
			ind = find(r == 0);
			r(ind) = -1;
		end
		M(:,t) = r';
		clear r ind
	    
		% compute distribution over (subset of) training examples
		DD = zeros(tn,1);
		for i = 1:tn
			for j = 1:noOfClasses
				ind = find(classLabels == train_labels(i));
				DD(i) = DD(i) + D(i,j) * double(M(ind,t) ~= M(j,t));
			end
		end
		DD = DD ./ sum(DD);
		    
		% Bi-partition dataset based on coloring
		colored_labels = zeros(size(train_labels,1),1);
	    
		for i = 1:noOfClasses
			ind = find(classLabels(i) == train_labels);
			if M(i,t) == 1
				colored_labels(ind) =  1;
			else
				colored_labels(ind) =  -1;
			end
		end
	    
		% learn classifier with distribution DD
		myLogInfo('[T%02d] learning %dth bi-partition (hash function)', trialNo, t);
	    
		if strcmp(opts.learner,'svm')
			% if libsvm gives an error, learn a decision tree
			try
				classifier(t).model = svmtrain(DD, colored_labels, double(train_data),...
				'-s 0 -t 0 -c 1000 -q');
				[outputLabels, accuracyVector, ~] = svmpredict(colored_labels,...
				double(train_data), classifier(t).model);
			catch
				myLogInfo('linear SVM error, learning non-linear svm')
				classifier(t).model = svmtrain(DD, colored_labels, double(train_data),...
				'-s 1 -t 2 -q');
				[outputLabels, accuracyVector, ~] = svmpredict(colored_labels,...
				double(train_data), classifier(t).model);
			end
		elseif strcmp(opts.learner,'stump')
			classifier(t).model = ClassificationTree.fit(double(train_data),colored_labels, ...
			'weights',DD,'minparent',length(DD));
			outputLabels = predict(classifier(t).model,double(train_data));
		else
			% Decision Tree
			classifier(t).model = ClassificationTree.fit(double(train_data),colored_labels,'weights',DD);
			outputLabels = predict(classifier(t).model,double(train_data));
		end
	    	
		we = sum((1 - (outputLabels == colored_labels)).*DD);

		% if classifier accuracy worse than random, flip the predictions
		if we < 0.5
			classifier(t).flip = 0;
		else
			classifier(t).flip = 1;
			outputLabels = -1 .* outputLabels;
			we = sum((1 - (outputLabels == colored_labels)).*DD);
		end
	    
		myLogInfo('[T%02d] %dth bi-partition weighted accuracy %3.2f',trialNo, t, 1-we);
		
		% re-weigh instances
		for i = 1:tn
			for j=1:noOfClasses
				setlabels = find(M(:,t) == outputLabels(i)); % this is a set
				ind = find(classLabels == train_labels(i)); % this is an index
				D(i,j) = D(i,j) * exp(opts.alpha * (~ismember(ind,setlabels) + ismember(j,setlabels)));
			end
		end
	    
		D(:,:) = D(:,:)./sum(sum(D));
	    
	end
	traintimes = toc;
	myLogInfo('[T%02d] Finished in %d sec, Saving model',trialNo, traintimes);

	% KH: save final model, etc
	F = [prefix '.mat'];
	save(F, 'classifier', 'traintimes','M');
	if ~opts.windows, unix(['chmod o-w ' F]); end % matlab permission bug
	myLogInfo('[T%02d] Saved: %s\n', trialNo, F);
end

	
function test_shecc(resfn, res_trial_fn, res_exist, opts)
	% if we're running this function, it means some elements in res_exist is false
	% and we need to compute/recompute the corresponding res_trial_fn's
	global Xtest Ytest Ytrain Xtrain

	testX  = Xtest;
	testY  = Ytest;
	trainY = Ytrain;
	trainX = Xtrain;
	[tn, td] = size(trainX);
	[tesn,tesd] = size(testX);
	
	
	[st, i] = dbstack();
	caller = st(2).name;

	% handle test_frac
	if opts.test_frac < 1
		myLogInfo('! only testing first %g%%', opts.test_frac*100);
		idx = 1:round(size(Xtest, 1)*opts.test_frac);
		testX = Xtest(idx, :);
		testY = Ytest(idx, :);
	end
	if size(Ytrain, 2) == 1
		trainY = floor(Ytrain/10);
		testY  = floor(Ytest/10);
		cateTrainTest = [];
	else
		cateTrainTest = (trainY * testY' > 0);
	end

	classLabels = unique(trainY);
	noOfClasses = length(classLabels);
	clear res train_time
	for t = 1:opts.ntrials
		if res_exist(t)
			myLogInfo('Trial %d: results exist', t);
			load(res_trial_fn{t});
		else
			clear t_res t_train_time
			Tprefix = sprintf('%s/trial%d', opts.expdir, t);
			trial_model = load(sprintf('%s.mat', Tprefix));
			Htrain = zeros(opts.nbits, tn ,'single');
			myLogInfo('[T%02d]:Generating hash codes for TRAIN',t);
			if strcmp(opts.mapping,'smooth')
		    		% map train data
			    	for j=1:opts.nbits
					myLogInfo('[T%02d]: Smooth mapping, generating %dth column bits', t, j);
					if strcmp(opts.learner,'svm')
				    		[output_labels, ~, ~] = svmpredict(ones(tn,1), double(trainX), trial_model.classifier(j).model);
					elseif strcmp(opts.learner,'stump') || strcmp(opts.learner,'tree')
						 output_labels = predict(trial_model.classifier(j).model,trainX);
					end
					Htrain(j,:) = output_labels';
					if trial_model.classifier(j).flip == 1
						Htrain(j,:) = -1 .* Htrain(j,:);
					end
				 end
			else
				myLogInfo('[T%02d]: Bucket mapping', t);
			try
				for i=1:noOfClasses
					ind = find(classLabels(i) == trainY);
					Htrain(:,ind) = repmat(trial_model.M(i,:)',1,length(ind));
				end
			catch ME	
				disp(ME.message);
				keyboard
			end
			end
			
			% map test data
			Htest = zeros(opts.nbits,tesn,'single');
			myLogInfo('[T%02d]:Generating hash codes for TEST',t);
			for j=1:opts.nbits
				myLogInfo('[T%02d]: Smooth mapping, generating %dth column bits', t, j);
				if strcmp(opts.learner,'svm')
					[output_labels, ~, ~] = svmpredict(ones(tesn,1), double(testX), trial_model.classifier(j).model);
				elseif strcmp(opts.learner,'stump') || strcmp(opts.learner,'tree')
					output_labels = predict(trial_model.classifier(j).model,double(testX));
				end
				Htest(j,:) = output_labels';
				if trial_model.classifier(j).flip == 1
					Htest(j,:) = -1 .* Htest(j,:);
				end
			end
			% get_results expects logical type 
			Htrain = logical((Htrain + 1)./2);
			Htest = logical((Htest + 1)./2);
			t_res(i) = get_results(Htrain, Htest, trainY, testY, opts, cateTrainTest);
			t_train_time(i) = trial_model.traintimes;
			clear Htrain Htest
			save(res_trial_fn{t}, 't_res', 't_train_time');
		end
		res(t, :) = t_res;
		train_time(t, :) = t_train_time;
	end
	myLogInfo('Final test %s: %.3g +/- %.3g', ...
		opts.metric, mean(res(:,end)), std(res(:,end)));

	% save all trials in a single file (for backward compatibility)
	% it may overwrite existing file, but whatever
	save(resfn, 'res', 'train_time');

end

