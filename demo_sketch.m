function resfn = demo_sketch(ftype, dataset, nbits, varargin)
	% PARAMS
	%  ftype (string) from {'gist', 'cnn'}
	%  dataset (string) from {'cifar', 'sun','nus'}
	%  nbits (integer) is length of binary code
	%  varargin: see get_opts.m for details
	%  
	addpath('sketch');
	opts = get_opts_sketch(ftype, dataset, nbits, varargin{:});  % set parameters

	% 0. result files
	Rprefix = sprintf('%s/%s', opts.expdir, opts.metric);
	if opts.test_frac < 1
		Rprefix = sprintf('%s_frac%g', Rprefix);
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
			if exist(res_trial_fn{t}, 'file')
				run_trial(t) = 0;
			else
				% [hack] for backward compatibility:
				% if final model trial_%d.mat exists and all the intermediate models 
				% exist as well, then we did this trial previously, just didn't save
				% res_trial_fn{t} -- NO need to rerun training
				modelprefix = sprintf('%s/trial%d', opts.expdir, t);
				try
					model = load([modelprefix '.mat']);
					model_exist = arrayfun(@(i) ...
						exist(sprintf('%s_iter%d.mat', modelprefix, i), 'file'), ...
						model.test_iters);
					if all(model_exist), run_trial(t) = 0; 
					else, run_trial(t) = 1; end
				catch
					run_trial(t) = 1;
				end
			end
		end
	end


	% 2. load data (only if necessary)
	global Xtrain Xtest Ytrain Ytest Dtype
	Dtype_this = [dataset '_' ftype];
	if ~isempty(Dtype) && strcmp(Dtype_this, Dtype)
		myLogInfo('Dataset already loaded for %s', Dtype_this);
	elseif any(run_trial) || ~all(res_exist)
		myLogInfo('Loading data for %s...', Dtype_this);
		eval(['[Xtrain, Ytrain, Xtest, Ytest] = load_' opts.ftype '(dataset, opts, 0);']);
		Dtype = Dtype_this;
	end

	% 3. TRAINING: run all _necessary_ trials (handled by train_osh)
	if any(run_trial)
		myLogInfo('Training models...');
		train_sketch(run_trial, opts);
	end
	myLogInfo('Training is done.');

	% 4. TESTING: run all _necessary_ trials
	if ~all(res_exist) || ~exist(resfn, 'file')
		myLogInfo('Testing models...');
		test_sketch(resfn, res_trial_fn, res_exist, opts);
	end
	myLogInfo('Testing is done.');
end



function opts = get_opts_sketch(ftype, dataset, nbits, varargin)
	% PARAMS
	%  mapping (string) {'smooth', 'bucket', 'coord'}
	%  ntrials (int) # of random trials
	%  stepsize (float) is step size in SGD
	%  SGDBoost (integer) is 0 for OSHEG, 1 for OSH
	%  randseed (int) random seed for repeatable experiments
	%  update_interval (int) update hash table
	%  test_interval (int) save/test intermediate model
	%  sampleratio (float) reservoir size, % of training set
	%  localdir (string) where to save stuff
	%  noTrainingPoints (int) # of training points 
	%  override (int) override previous results {0, 1}
	% 
	ip = inputParser;
	% default values
	ip.addParamValue('ftype', ftype, @isstr);
	ip.addParamValue('dataset', dataset, @isstr);
	ip.addParamValue('nbits', nbits, @isscalar);

	ip.addParamValue('mapping', 'smooth', @isstr);
	ip.addParamValue('randseed', 12345, @isscalar);
	ip.addParamValue('localdir', ...
		'/research/object_detection/cachedir/online-hashing/sketch', @isstr);
	ip.addParamValue('noTrainingPoints', 20000, @isscalar);
	ip.addParamValue('override', 0, @isscalar);
	ip.addParamValue('showplots', 1, @isscalar);

	ip.addParamValue('nworkers', 6, @isscalar);
	ip.addParamValue('ntrials', 5, @isscalar);
	ip.addParamValue('ntests', 20, @isscalar);  % <1 for faster testing
	ip.addParamValue('test_frac', 1, @isscalar);  % <1 for faster testing
	ip.addParamValue('metric', 'mAP', @isstr);    % evaluation metric

	% controling when to update hash table
	%ip.addParamValue('update_interval', -1, @isscalar);  % use with baseline

	% Hack for Places
	ip.addParamValue('labelspercls', 0, @isscalar);

	% NOTE specific for online sketching hashing
	ip.addParamValue('sketchsize', 200, @isscalar);
	ip.addParamValue('batchsize', 50, @isscalar);
	ip.addParamValue('onlyfinal', 0, @isscalar);
	
	% parse input
	ip.parse(varargin{:});
	opts = ip.Results;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% assertions
	assert(ismember(opts.ftype, {'gist', 'cnn'}));
	assert(opts.test_frac > 0);
	assert(opts.nworkers>0 && opts.nworkers<=12);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	% are we on window$?
	opts.windows = ~isempty(strfind(computer, 'WIN'));
	if opts.windows
		% reset localdir
		opts.localdir = '\\kraken\object_detection\cachedir\online-hashing\sketch';
		myLogInfo('We are on Window$. localdir set to %s', opts.localdir);
	end

	% matlabpool handling
	if matlabpool('size') == 0
		myLogInfo('Opening matlabpool, nworkers = %d', opts.nworkers);
		matlabpool close force local  % clear up zombies
		matlabpool(opts.nworkers);
	end

	% make localdir
	if ~exist(opts.localdir, 'dir'), 
		mkdir(opts.localdir);  
		if ~opts.windows, unix(['chmod g+rw ' opts.localdir]); end
	end

	% set randseed -- don't change the randseed if don't have to!
	rng(opts.randseed);

	% [hack] for places
	if strcmp(opts.dataset, 'places')
		if opts.labelspercls > 0
			assert(opts.labelspercls >= 500 && opts.labelspercls <= 5000, ...
				'please give a reasonable labelspercls in [500, 5000]');
			myLogInfo('Places will use %d labeled examples per class', opts.labelspercls);
			opts.dataset = [opts.dataset, 'L', num2str(opts.labelspercls)];
		else
			myLogInfo('Places: fully supervised experiment');
		end
	end

	% identifier string for the current experiment
	opts.identifier = sprintf('%s-%s-%d%s-Ske%dBat%d-%dpts-%dtests', opts.dataset, opts.ftype, ...
		opts.nbits, opts.mapping, opts.sketchsize, opts.batchsize, opts.noTrainingPoints, opts.ntests);
	if opts.onlyfinal
		opts.identifier = [opts.identifier, '-final'];
	end
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

	% FINISHED
	disp(opts);
end
