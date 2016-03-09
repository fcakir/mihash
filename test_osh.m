function test_osh(Xtest, Ytest, cateTrainTest, resfn, res_trial_fn, opts)
	try 
		load(resfn);
	catch
		clear res bitflips train_iter train_time
		for t = 1:opts.ntrials
			try
				load(res_trial_fn{t});
			catch
				clear t_res t_bitflips t_train_iter t_train_time
				Tprefix = sprintf('%s/trial%d', opts.expdir, t);
				trial_model = load(sprintf('%s.mat', Tprefix));
				for i = 1:opts.ntests
					iter = trial_model.test_iters(i);
					d = load(sprintf('%s_iter%d.mat', Tprefix, iter));
					Y = d.H;  % NOTE: logical
					tY = (d.W'*Xtest' > 0);

					% NOTE: get_res() uses parfor
					fprintf('Trial %d, Iter %5d/%d, ', t, iter, opts.noTrainingPoints);
					t_res(i) = get_results(cateTrainTest, Y, tY, opts);
					t_bitflips(i) = d.bitflips;
					t_train_iter(i) = iter;
					t_train_time(i) = d.train_time;
				end
				save(res_trial_fn{t}, 't_res', 't_bitflips', 't_train_iter', 't_train_time');
			end
			res(t, :) = t_res;
			bitflips(t, :) = t_bitflips;
			train_iter(t, :) = t_train_iter;
			train_time(t, :) = t_train_time;
		end
		% save all trials in a single file (for backward compatibility)
		save(resfn, 'res', 'bitflips', 'train_iter', 'train_time');
	end
	myLogInfo('Final test %s: %.3g +/- %.3g', opts.metric, mean(res(:,end)), std(res(:,end)));

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
