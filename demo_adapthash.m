function resfn = demo_adapthash(ftype, dataset, nbits, varargin)

	opts = get_opts_adapthash(ftype, dataset, nbits, varargin{:});  % set parameters

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
	elseif (any(run_trial) || ~all(res_exist))
		myLogInfo('Loading data for %s...', Dtype_this);
		eval(['[Xtrain, Ytrain, Xtest, Ytest] = load_' opts.ftype '(dataset, opts);']);
		Dtype = Dtype_this;

		% DEBUG
		if opts.debug
			ind = randperm(size(Xtrain, 1), 10000);
			Xtrain = Xtrain(ind, :);
			Ytrain = Ytrain(ind);
		end
		% DEBUG
	end

	% 3. TRAINING: run all _necessary_ trials (handled by train_osh)
	if any(run_trial)
		myLogInfo('Training models...');
		train_adapthash(run_trial, opts);
	end
	myLogInfo('Training is done.');

	% 4. TESTING: run all _necessary_ trials
	if ~all(res_exist) || ~exist(resfn, 'file')
		% NOTE reusing test_osh for AdaptHash
		myLogInfo('Testing models...');
		myLogInfo('<<NOTE>> AdaptHash uses pairs, so each iteration uses 2 examples');
		opts.noTrainingPoints = opts.noTrainingPoints / 2;
		test_osh(resfn, res_trial_fn, res_exist, opts);
	end
	myLogInfo('Testing is done.');
end



function opts = get_opts_adapthash(ftype, dataset, nbits, varargin)

	ip = inputParser;

	% default values
	ip.addParamValue('ftype', ftype, @isstr);
	ip.addParamValue('dataset', dataset, @isstr);
	ip.addParamValue('nbits', nbits, @isscalar);

	ip.addParamValue('debug', 0, @isscalar);

	ip.addParamValue('nworkers', 6, @isscalar);
	ip.addParamValue('randseed', 12345, @isscalar);

	ip.addParamValue('override', 0, @isscalar);
	ip.addParamValue('ntrials', 5, @isscalar);
	ip.addParamValue('noTrainingPoints', 2000, @isscalar);
	ip.addParamValue('mapping', 'smooth', @isstr);

	ip.addParamValue('ntests', 20, @isscalar);
	ip.addParamValue('metric', 'mAP', @isstr);    % evaluation metric
	ip.addParamValue('test_frac', 1, @isscalar);  % <1 for faster testing
	ip.addParamValue('showplots', 1, @isscalar);

	% controling when to update hash table
	ip.addParamValue('update_interval', -1, @isscalar);  % use with baseline

	% Hack for Places
	ip.addParamValue('labelspercls', 0, @isscalar);
	
	% AdaptHash-specific
	ip.addParamValue('localdir', ...
		'/research/object_detection/cachedir/online-hashing/adapt', @isstr);
	ip.addParamValue('alpha', 0.9, @isscalar);
	ip.addParamValue('beta', 1e-2, @isscalar);
	ip.addParamValue('stepsize', 1, @isscalar);

	% parse input
	ip.parse(varargin{:});
	opts = ip.Results;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% assertions
	assert(ismember(opts.ftype, {'gist', 'cnn'}));
	assert(strcmp(opts.mapping, 'smooth'), 'only doing SMOOTH mapping for now');
	assert(opts.test_frac > 0);
	assert(opts.nworkers>0 && opts.nworkers<=12);
	assert(mod(opts.noTrainingPoints, 2)==0);
	if opts.update_interval > 0
		assert(mod(opts.update_interval, 2) == 0);
	end

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
	opts.identifier = sprintf('%s-%s-%d%s-A%gB%gS%g', opts.dataset, opts.ftype, ...
		opts.nbits, opts.mapping, opts.alpha, opts.beta, opts.stepsize);
	if opts.update_interval > 0
		opts.identifier = sprintf('%s-U%d', opts.identifier, opts.update_interval);
	else
		opts.ntests = 2;
	end
	opts.identifier = sprintf('%s-%dpts-%dtests', opts.identifier, ...
		opts.noTrainingPoints, opts.ntests);
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
