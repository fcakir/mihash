function osh_gist(dataset, nbits, varargin)
	% PARAMS
	%  dataset (string) from {'cifar' ,'sun','nus'}
	%  nbits (integer) is length of binary code
	%  varargin: see get_opts.m for details
	%  
	if nargin < 1, dataset = 'cifar'; end
	if nargin < 2, nbits = 8; end
	opts = get_opts(dataset, nbits, varargin{:});  % set parameters

	%mAPfn = sprintf('%s/mAP_t%d.mat', opts.expdir, opts.test_interval);
	mAPfn = sprintf('%s/mAP_%dtests', opts.expdir, opts.ntests);
	if opts.test_frac < 1
		mAPfn = sprintf('%s_frac%g', mAPfn);
	end
	if opts.override==0 && exist([mAPfn '.mat'], 'file')
		% load experiment results
		load([mAPfn '.mat']);
		myLogInfo(['Results loaded: ' mAPfn]);
	else
		% load GIST data
		[traingist, trainlabels, testgist, testlabels, cateTrainTest, opts] = ...
			load_gist(dataset, opts);
		if opts.test_frac < 1
			myLogInfo('! only testing first %g%%', opts.test_frac*100);
			idx = 1:round(size(testgist, 1)*opts.test_frac);
			testgist = testgist(idx, :);
			cateTrainTest = cateTrainTest(:, idx);
		end

		% ONLINE LEARNING
		train_osh(traingist, trainlabels, opts);
		
		% test models
		n          = opts.ntests;
		mAP        = zeros(opts.ntrials, n);
		bitflips   = zeros(opts.ntrials, n);
		train_time = zeros(opts.ntrials, n);
		for t = 1:opts.ntrials
			trial_model = load(sprintf('%s/trial%d.mat', opts.expdir, t));
			for i = 1:trial_model.test_iters
				F = sprintf('%s/trial%d_iter%d.mat', opts.expdir, t, i*opts.test_interval);
				d = load(F);
				W = d.W;
				Y = d.Y;
				tY = 2*single(W'*testgist' > 0)-1;
		
				% NOTE: get_mAP() uses parfor
				fprintf('Trial %d, Iter %d, ', t, i);
				mAP(t, i) = get_mAP(cateTrainTest, Y, tY);
				bitflips(t, i) = d.bitflips;
				train_time(t, i) = d.train_time;
			end
		end
		save([mAPfn '.mat'], 'mAP', 'bitflips', 'train_time');
		myLogInfo(['Results saved: ' mAPfn]);
	end
	myLogInfo('Test mAP (final): %.3g +/- %.3g', mean(mAP(:,end)), std(mAP(:,end)));

	% draw curves
	[px, py] = avg_curve(mAP, bitflips);
	figure, if length(px) == 1, plot(px, py,'+'), else plot(px,py), end, grid, title(opts.identifier)
	xlabel('bit flips'), ylabel('mAP')

	[px, py] = avg_curve(mAP, train_time);
	figure, if length(px) == 1, plot(px, py,'+'), else plot(px,py), end, grid, title(opts.identifier)
	xlabel('CPU time'), ylabel('mAP')
end
