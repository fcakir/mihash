function test_osh(resfn, res_trial_fn, res_exist, opts)
	% if we're running this function, it means some elements in res_exist is false
	% and we need to compute/recompute the corresponding res_trial_fn's
	global Xtest Ytest Ytrain

	% handle test_frac
	if opts.test_frac < 1
		myLogInfo('! only testing first %g%%', opts.test_frac*100);
		idx = 1:round(size(Xtest, 1)*opts.test_frac);
		testX = Xtest(idx, :);
		testY = Ytest(idx);
	else
		testX = Xtest;
		testY = Ytest;
	end

	% for semi-supervised case, only do retrieval against LABELED training data
	% NOTE assuming single-labeled examples
	if ~all(Ytrain > 0)
		labeled = find(Ytrain > 0);
		myLogInfo('Doing retrieval against the %d(%.1f%%) labeled training examples', ...
			length(labeled), length(labeled)/length(Ytrain)*100);
		Ytrain  = Ytrain(labeled);
	else
		labeled = [];
	end

	clear res bitflips train_iter train_time
	for t = 1:opts.ntrials
		if res_exist(t)
			myLogInfo('Trial %d: results exist', t);
			load(res_trial_fn{t});
		else
			clear t_res t_bitflips t_train_iter t_train_time
			Tprefix = sprintf('%s/trial%d', opts.expdir, t);
			trial_model = load(sprintf('%s.mat', Tprefix));
			for i = 1:length(trial_model.test_iters)  % may NOT be 1:opts.ntests!
				iter = trial_model.test_iters(i);
				d = load(sprintf('%s_iter%d.mat', Tprefix, iter));
				if isempty(labeled)
					Htrain = d.H;
				else
					Htrain = d.H(:, labeled);
				end
				if 0 %~isempty(d.seenLabels)
					[~, ind] = ismember(d.seenLabels, Ytest);
					Htest = (d.W'*testX(ind, :)' > 0);
					Ltest = testY(ind);
				else
					Htest = (d.W'*testX' > 0);
					Ltest = testY;
				end

				fprintf('Trial %d, Iter %5d/%d, ', t, iter, opts.noTrainingPoints);
				t_res(i) = get_results(Htrain, Htest, Ytrain, Ltest, opts);
				t_bitflips(i) = d.bitflips;
				t_train_iter(i) = iter;
				t_train_time(i) = d.train_time;
			end
			clear Htrain Htest Ltest
			save(res_trial_fn{t}, 't_res', 't_bitflips', 't_train_iter', 't_train_time');
		end
		res(t, :) = t_res;
		bitflips(t, :) = t_bitflips;
		train_iter(t, :) = t_train_iter;
		train_time(t, :) = t_train_time;
	end
	myLogInfo('Final test %s: %.3g +/- %.3g', ...
		opts.metric, mean(res(:,end)), std(res(:,end)));

	% save all trials in a single file (for backward compatibility)
	% it may overwrite existing file, but whatever
	save(resfn, 'res', 'bitflips', 'train_iter', 'train_time');

	% visualize
	if opts.showplots
		% draw curves, with auto figure saving
		figname = sprintf('%s_iter.fig', resfn);
		show_res(figname, res, train_iter, 'iterations', opts.identifier);
		figname = sprintf('%s_cpu.fig', resfn);
		show_res(figname, res, train_time, 'CPU time', opts.identifier);
		figname = sprintf('%s_flip.fig', resfn);
		show_res(figname, res, bitflips, 'bit flips', opts.identifier);
		drawnow;
	end
end

% -----------------------------------------------------------
function show_res(figname, Y, X, xlb, ttl)
	try
		openfig(figname);
	catch
		[px, py] = avg_curve(Y, X);
		figure, if length(px) == 1, plot(px, py, '+'), else plot(px, py), end
		grid, title(ttl), xlabel(xlb), ylabel('res')
		saveas(gcf, figname);
	end
end
