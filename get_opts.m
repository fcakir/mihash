function opts = get_opts(dataset, nbits, varargin)
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
	ip.addParamValue('dataset', dataset, @isstr);
	ip.addParamValue('nbits', nbits, @isscalar);
	ip.addParamValue('mapping', 'smooth', @isstr);
	ip.addParamValue('ntrials', 10, @isscalar);
	ip.addParamValue('stepsize', 0.2, @isscalar);
	ip.addParamValue('SGDBoost', 0, @isscalar);
	ip.addParamValue('randseed', 12345, @isscalar);
	ip.addParamValue('localdir', '/scratch/online-hashing', @isstr);

	ip.addParamValue('noTrainingPoints', 2000, @isscalar);
	ip.addParamValue('override', 0, @isscalar);

	% controling when to update hash table
	% default: save every opts.update_interval iterations
	% IF use reservoir AND opts.flip_thresh > 0, THEN use opts.flip_thresh
	ip.addParamValue('update_interval', 200, @isscalar); % use with baseline
	ip.addParamValue('flip_thresh', 10, @isscalar); % use with reservoir

	% testing
	ip.addParamValue('test_interval', -1, @isscalar);   % save intermediate model
	ip.addParamValue('test_frac', 1, @isscalar);         % for faster testing
	ip.addParamValue('ntests', 10, @isscalar);
	
	%ip.addParamValue('exp', 'baseline', @isstr);   % baseline, rs, l1l2
	ip.addParamValue('samplesize', 200, @isscalar); % reservoir size
	ip.addParamValue('reg_rs', -1, @isscalar);      % reservoir reg. weight
	ip.addParamValue('reg_maxent', -1, @isscalar);  % max entropy reg. weight
	ip.addParamValue('reg_smooth', -1, @isscalar);  % smoothness reg. weight
	
	% parse input
	ip.parse(varargin{:});
	opts = ip.Results;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	% assertions
	assert(~(opts.reg_maxent>0 && opts.reg_smooth>0));  % can't have both
	assert(opts.test_frac > 0);
	assert(opts.test_interval <= opts.noTrainingPoints, ... 
		'test_interval should be smaller than \# of training points');

	% make localdir
	if ~exist(opts.localdir, 'dir'), 
		mkdir(opts.localdir); 
		unix(['chmod g+rw ' opts.localdir]);
	end

	% set randseed -- don't change the randseed if don't have to!
	rng(opts.randseed);

	% identifier string for the current experiment
	opts.identifier = sprintf('%s-%d%s-B%dst%g', opts.dataset, opts.nbits, ...
		opts.mapping, opts.SGDBoost, opts.stepsize, opts.ntests);
	if opts.reg_rs > 0
		assert(opts.flip_thresh > 0);
		opts.identifier = sprintf('%s-RS%dL%gF%g', opts.identifier, ...
			opts.samplesize, opts.reg_rs, opts.flip_thresh);
	else
		assert(opts.update_interval > 0);
		opts.identifier = sprintf('%s-U%d', opts.identifier, opts.update_interval);
	end
	if opts.reg_maxent > 0
		opts.identifier = sprintf('%s-ME%g', opts.identifier, opts.reg_maxent);
	end
	if opts.reg_smooth > 0
		; %TODO
	end

	% set expdir
	opts.expdir = sprintf('%s/%s', opts.localdir, opts.identifier);
	if ~exist(opts.expdir, 'dir'), 
		myLogInfo(['creating opts.expdir: ' opts.expdir]);
		mkdir(opts.expdir); unix(['chmod g+rw ' opts.expdir]); 
	end

	% FINISHED
	disp(opts);
end
