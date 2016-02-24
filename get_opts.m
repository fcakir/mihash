function opts = get_opts(dataset, nbits, varargin)
	% PARAMS
	%  mapping: {'smooth', 'bucket', 'coord'}
	%  ntrials: # of random trials
	%  stepsize (float) is step size in SGD
	%  SGDBoost (integer) is 0 for OSHEG, 1 for OSH
	%  randseed: (int) random seed for repeatable experiments
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
	ip.addParamValue('update_interval', 50, @isscalar);  % update index structure
	ip.addParamValue('test_interval', 200, @isscalar);  % save intermediate model
	ip.addParamValue('sampleratio', 0.01, @isscalar);  % reservoir size
	ip.addParamValue('localdir', '/scratch/online-hashing', @isstr);
	ip.addParamValue('exp', 'baseline', @isstr);  % baseline, rs, l1l2
	% parse input
	ip.parse(varargin{:});
	opts = ip.Results;

	% make localdir
	if ~exist(opts.localdir, 'dir'), 
		mkdir(opts.localdir); 
		unix(['chmod g+rw ' opts.localdir]);
	end

	% set randseed
	rng(opts.randseed);

	% identifier string for the current experiment
	opts.identifier = sprintf('%s-%dbit-%s-r%d-st%g', ...
		opts.dataset, opts.nbits, opts.mapping, opts.randseed, opts.stepsize);

	% set expdir
	if strcmp(opts.exp, 'baseline')
		expdir = sprintf('%s/%s-u%d', opts.localdir, opts.identifier, opts.update_interval);
	elseif strcmp(opts.exp, 'rs')
		expdir = sprintf('%s/%s-u%d-RS', opts.localdir, opts.identifier, opts.update_interval);
	elseif strcmp(opts.exp, 'l1l2')
		expdir = sprintf('%s/%s-u%d-L1L2', opts.localdir, opts.identifier, opts.update_interval);
	end
	if ~exist(expdir, 'dir'), 
		myLogInfo(['creating expdir: ' expdir]);
		mkdir(expdir); unix(['chmod g+rw ' expdir]); 
	end

	% FINISHED
	disp(opts);
end
