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
	ip.addParamValue('stepsize', 0.1, @isscalar);
	ip.addParamValue('SGDBoost', 0, @isscalar);
	ip.addParamValue('randseed', 12345, @isscalar);
	ip.addParamValue('update_interval', 50, @isscalar);  % update index structure
	ip.addParamValue('test_interval', 200, @isscalar);  % save intermediate model
	ip.addParamValue('sampleratio', 0.01, @isscalar);  % reservoir size
	ip.addParamValue('localdir', '/scratch/online-hashing', @isstr);
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
	opts.identifier = sprintf('%s-%dbit-%s-r%d', ...
		opts.dataset, opts.nbits, opts.mapping, opts.randseed);

	% FINISHED
	disp(opts);
end
