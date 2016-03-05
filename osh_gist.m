function osh_gist(dataset, nbits, varargin)
	% PARAMS
	%  dataset (string) from {'cifar' ,'sun','nus'}
	%  nbits (integer) is length of binary code
	%  varargin: see get_opts.m for details
	%  
	if nargin < 1, dataset = 'cifar'; end
	if nargin < 2, nbits = 8; end
	opts = get_opts(dataset, nbits, varargin{:});  % set parameters

	mAPfn = sprintf('%s/mAP_%dtrials', opts.expdir, opts.ntrials);
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
		train_iter = zeros(opts.ntrials, n);
		train_time = zeros(opts.ntrials, n);
		for t = 1:opts.ntrials
			prefix = sprintf('%s/trial%d', opts.expdir, t);
			trial_model = load(sprintf('%s.mat', prefix));
			for i = 1:n
				iter = trial_model.test_iters(i);
				d = load(sprintf('%s_iter%d.mat', prefix, iter));
				Y = d.H;  % NOTE: logical
				tY = (d.W'*testgist' > 0);
		
				% NOTE: get_mAP() uses parfor
				fprintf('Trial %d, Iter %5d/%d, ', t, iter, opts.noTrainingPoints);
				mAP(t, i) = get_mAP(cateTrainTest, Y, tY);
				bitflips(t, i) = d.bitflips;
				train_iter(t, i) = iter;
				train_time(t, i) = d.train_time;
			end
		end
		save([mAPfn '.mat'], 'mAP', 'bitflips', 'train_time', 'train_iter');
		myLogInfo(['Results saved: ' mAPfn]);
	end
	myLogInfo('Test mAP (final): %.3g +/- %.3g', mean(mAP(:,end)), std(mAP(:,end)));

	% draw curves, with auto figure saving
	figname = sprintf('%s_iter.fig', mAPfn);
	show_mAP(figname, mAP, train_iter, 'iterations', opts.identifier);

	figname = sprintf('%s_cpu.fig', mAPfn);
	show_mAP(figname, mAP, train_time, 'CPU time', opts.identifier);

	figname = sprintf('%s_flip.fig', mAPfn);
	show_mAP(figname, mAP, bitflips, 'bit flips', opts.identifier);
end

% -----------------------------------------------------------
function show_mAP(figname, mAP, X, xlb, ttl)
	try
		openfig(figname);
	catch
		[px, py] = avg_curve(mAP, X);
		figure, if length(px) == 1, plot(px, py,'+'), else plot(px,py), end
		grid, title(ttl), xlabel(xlb), ylabel('mAP')
		saveas(gcf, figname);
	end
end
