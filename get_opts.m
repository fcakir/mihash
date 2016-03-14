function opts = get_opts(ftype, dataset, nbits, varargin)
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
	ip.addParamValue('stepsize', 0.1, @isscalar);
	ip.addParamValue('SGDBoost', 0, @isscalar);
	ip.addParamValue('randseed', 12345, @isscalar);
	ip.addParamValue('localdir', ...
		'/research/object_detection/cachedir/online-hashing', @isstr);
	ip.addParamValue('noTrainingPoints', 20000, @isscalar);
	ip.addParamValue('override', 0, @isscalar);
	ip.addParamValue('showplots', 1, @isscalar);

	ip.addParamValue('nworkers', 6, @isscalar);
	ip.addParamValue('ntrials', 5, @isscalar);
	ip.addParamValue('ntests', 10, @isscalar);
	ip.addParamValue('test_frac', 1, @isscalar);  % <1 for faster testing
	ip.addParamValue('metric', 'mAP', @isstr);    % evaluation metric

	% controling when to update hash table
	% default: save every opts.update_interval iterations
	% IF use reservoir AND opts.flip_thresh > 0, THEN use opts.flip_thresh
	ip.addParamValue('update_interval', -1, @isscalar);  % use with baseline
	ip.addParamValue('flip_thresh', -1, @isscalar);      % use with reservoir
	ip.addParamValue('adaptive', -1, @isscalar);         % use with reservoir

	ip.addParamValue('samplesize', 50, @isscalar);    % reservoir size
	ip.addParamValue('reg_rs', -1, @isscalar);        % reservoir reg. weight
	ip.addParamValue('reg_maxent', -1, @isscalar);    % max entropy reg. weight
	ip.addParamValue('reg_smooth', -1, @isscalar);    % smoothness reg. weight
	ip.addParamValue('rs_sm_neigh_size',2,@isscalar); % neighbor size for smoothness

	% Hack for Places
	ip.addParamValue('labelspercls', 0, @isscalar);
	
	% parse input
	ip.parse(varargin{:});
	opts = ip.Results;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% assertions
	assert(ismember(opts.ftype, {'gist', 'cnn'}));
	assert(~(opts.reg_maxent>0 && opts.reg_smooth>0));  % can't have both
	assert(opts.test_frac > 0);
	assert(opts.ntests >= 2, 'ntests should be at least 2 (first & last iter)');
	assert(~(opts.update_interval>0 && (opts.flip_thresh>0 || opts.adaptive>0)), ...
		'update_interval cannot be used with flip_thresh or adaptive');
	if opts.adaptive > 0, 
		assert(opts.flip_thresh<=0, 'adaptive cannot have flip_thresh>0'); 
	end

	if ~strcmp(opts.mapping,'smooth')
		opts.update_interval = opts.noTrainingPoints;
		myLogInfo([opts.mapping ' hashing scheme supports ntests = 2 only' ...
			'\n setting ntests to 2'])
		opts.ntests = 2;
	end

	assert(opts.nworkers>0 && opts.nworkers<=12);

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	% are we on window$?
	opts.windows = ~isempty(strfind(computer, 'WIN'));
	if opts.windows
		% reset localdir
		opts.localdir = '\\kraken\object_detection\cachedir\online-hashing';
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

	% FC: if mapping is not smooth, set update_interval to noTrainingPoints
	if ~strcmp(opts.mapping, 'smooth') && opts.update_interval > 0 && ...
			(opts.update_interval ~= opts.noTrainingPoints)
		myLogInfo('Mapping: %s. Overriding update_interval=%d to noTrainingPoints=%d', ...
			opts.mapping, opts.update_interval, opts.noTrainingPoints);
		opts.update_interval = opts.noTrainingPoints;
	end

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
	opts.identifier = sprintf('%s-%s-%d%s-B%dS%g', opts.dataset, opts.ftype, ...
		opts.nbits, opts.mapping, opts.SGDBoost, opts.stepsize);
	if opts.reg_rs > 0
		% reservoir: use update_interval or flip_thresh or adaptive
		if opts.update_interval > 0
			opts.identifier = sprintf('%s-RS%dL%gU%g', opts.identifier, ...
				opts.samplesize, opts.reg_rs, opts.update_interval);
		elseif opts.flip_thresh > 0
			opts.identifier = sprintf('%s-RS%dL%gF%g', opts.identifier, ...
				opts.samplesize, opts.reg_rs, opts.flip_thresh);
		else
			assert(opts.adaptive > 0);
			opts.identifier = sprintf('%s-RS%dL%gAda', opts.identifier, ...
				opts.samplesize, opts.reg_rs);
		end
	else
		% no reservoir (baseline): use update_interval
		assert(opts.update_interval > 0);
		opts.identifier = sprintf('%s-U%d', opts.identifier, opts.update_interval);
	end
	if opts.reg_maxent > 0
		opts.identifier = sprintf('%s-ME%g', opts.identifier, opts.reg_maxent);
	end
	if opts.reg_smooth > 0
		opts.identifier = sprintf('%s-SM%gN%d', opts.identifier, opts.reg_smooth, opts.rs_sm_neigh_size);
	end
	myLogInfo('identifier: %s', opts.identifier);

	% set expdir
	expdir_base = sprintf('%s/%s', opts.localdir, opts.identifier);
	opts.expdir = sprintf('%s/%gpts_%dtests', expdir_base, opts.noTrainingPoints, opts.ntests);
	if ~exist(expdir_base, 'dir'), 
		mkdir(expdir_base);
		if ~opts.windows, unix(['chmod g+rw ' expdir_base]); end
	end
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
