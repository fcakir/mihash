function resfn = demo_osh(ftype, dataset, nbits, varargin)
	% PARAMS
	%  ftype (string) from {'gist', 'cnn'}
	%  dataset (string) from {'cifar', 'sun','nus'}
	%  nbits (integer) is length of binary code
	%  varargin: see get_opts.m for details
	%  
	opts = get_opts(ftype, dataset, nbits, varargin{:});  % set parameters

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
	res_exist = cellfun(@(r) exist(r, 'file'), res_trial_fn);


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
	if any(run_trial) || ~all(res_exist)
		eval(['[Xtrain, Ytrain, Xtest, Ytest, cateTrainTest] = load_', opts.ftype, ...
			'(dataset, opts);']);

		% handle test_frac
		if opts.test_frac < 1
			myLogInfo('! only testing first %g%%', opts.test_frac*100);
			idx = 1:round(size(Xtest, 1)*opts.test_frac);
			Xtest = Xtest(idx, :);
			cateTrainTest = cateTrainTest(:, idx);
		end
	end

	% 3. TRAINING: run all _necessary_ trials (handled by train_osh)
	if any(run_trial)
		train_osh(Xtrain, Ytrain, run_trial, opts);
	end

	% 4. TESTING: run all _necessary_ trials
	if ~all(res_exist)
		test_osh(Xtest, Ytest, cateTrainTest, resfn, res_trial_fn, opts);
	end
end
