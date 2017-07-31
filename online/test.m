function test(resfn, res_trial_fn, res_exist, opts)
% Computes the performance by loading and evaluating the "checkpoint" files
% saved during training. 
%
% INPUTS
% 	resfn - (string)Path to final results file.
%res_trial_fn - (cell) 	Cell array containing the paths to individual trial results
% 			files.
% res_exist   - (int)   Boolean vector indicating whether the trial results files needed
% 			to be computed. For example, if opts.override=0 and the
% 		   	corresponding trial result file is computed (from a previous
% 			but identical experiment), the testing is skipped for that
% 			trial.
%	opts  - (struct)Parameter structure.
% 
% OUTPUTS
%	none

global Xtest Ytest Xtrain Ytrain thr_dist

testX  = Xtest;
testY  = Ytest;
trainY = Ytrain;

[st, i] = dbstack();
caller = st(2).name;

% handle testFrac
if opts.testFrac < 1
    myLogInfo('! only testing first %g%%', opts.testFrac*100);
    idx = 1:round(size(Xtest, 1)*opts.testFrac);
    testX = Xtest(idx, :);
    testY = Ytest(idx, :);
end
if size(Ytrain, 2) == 1
    trainY = floor(Ytrain/10);
    testY  = floor(Ytest/10);
    cateTrainTest = [];
elseif size(Ytrain, 2) > 1
    cateTrainTest = (trainY * testY' > 0);
elseif isempty(Ytrain)
    cateTrainTest = pdist2(Xtrain, Xtest, 'euclidean') <= thr_dist; %logical
else
    error('Ytrain error in test.m');
end

clear res bitflips bits_computed_all
clear train_iter train_time train_examples

for t = 1:opts.ntrials
    if res_exist(t)
        myLogInfo('Trial %d: results exist', t);
        load(res_trial_fn{t});
    else
        clear t_res t_bits_computed_all t_bitflips
        clear t_train_iter t_train_time

        Tprefix = sprintf('%s/trial%d', opts.expdir, t);
        Tmodel = load(sprintf('%s.mat', Tprefix));

        % handle transformations to X
        if strcmp(opts.methodID, 'okh')
            % do kernel mapping for test data
            testX_t = exp(-0.5*sqdist(testX', Tmodel.Xanchor')/Tmodel.sigma^2)';
            testX_t = [testX_t; ones(1,size(testX_t,2))]';
        elseif strcmp(opts.methodID, 'sketch')
            % subtract mean
            testX_t = bsxfun(@minus, testX, Tmodel.instFeatAvePre);
        else
            % OSH, AdaptHash: nothing
            testX_t = testX;
        end

        for i = 1:length(Tmodel.test_iters)
            % determine whether to actually run test or not
            % if there's no HT update since last test, just copy results
            % THIS SAVES TIME!
            if i == 1
                runtest = true;
            else
                st = Tmodel.test_iters(i-1);
                ed = Tmodel.test_iters(i);
                runtest = any(Tmodel.update_iters>st & Tmodel.update_iters<=ed);
            end

            iter = Tmodel.test_iters(i);
            d = load(sprintf('%s_iter%d.mat', Tprefix, iter));
            fprintf('Trial %d, Checkpoint %5d/%d, ', t, iter*opts.batchSize, opts.noTrainingPoints*opts.epoch);

            if runtest
                Htrain = d.H;

                % test hash table
                % NOTE: for intermediate iters, need to use W_lastupdate (not W!)
                %       to compute Htest, to make sure it's computed using the same
                %       hash mapping as Htrain.
                Htest = (testX_t * d.W_lastupdate > 0)';

                % evaluate
                t_res(i) = evaluate(Htrain, Htest, trainY, testY, opts, cateTrainTest);

                t_bits_computed_all(i) = d.bits_computed_all;
                t_bitflips(i) = d.bitflips;
            else
                t_res(i) = t_res(i-1);
                t_bits_computed_all(i) = t_bits_computed_all(i-1);
                t_bitflips(i) = t_bitflips(i-1);
                fprintf(' %g\n', t_res(i));
            end
            t_train_time(i) = d.train_time;
            t_train_iter(i) = iter;
        end
        clear Htrain Htest
        save(res_trial_fn{t}, 't_res', 't_bitflips', 't_bits_computed_all', ...
            't_train_iter', 't_train_time');
        if ~opts.windows, 
            unix(['chmod g+w ' res_trial_fn{t}]); 
            unix(['chmod o-w ' res_trial_fn{t}]); 
        end
    end
    res(t, :) = t_res;
    bitflips(t, :) = t_bitflips;
    bits_computed_all(t, :) = t_bits_computed_all;
    train_time(t, :) = t_train_time;
    train_iter(t, :) = t_train_iter;
    train_examples(t, :) = t_train_iter * opts.batchSize;
end
myLogInfo('  FINAL %s: %.3g +/- %.3g', opts.metric, mean(res(:,end)), std(res(:,end)));
myLogInfo('    AUC %s: %.3g +/- %.3g', opts.metric, mean(mean(res, 2)), std(mean(res, 2)));

% save all trials in a single file (for backward compatibility)
% it may overwrite existing file, but whatever
save(resfn, 'res', 'bitflips', 'bits_computed_all', 'train_iter', 'train_time', ...
    'train_examples');
if ~opts.windows, unix(['chmod g+w ' resfn]); unix(['chmod o-w ' resfn]); end

% visualize
if opts.showplots
    % draw curves, with auto figure saving
    figname = sprintf('%s_trex.fig', resfn);
    show_res(figname, res, train_examples, 'Training Examples', opts.identifier, opts.override);
    figname = sprintf('%s_cpu.fig', resfn);
    show_res(figname, res, train_time, 'CPU Time', opts.identifier, opts.override);
    figname = sprintf('%s_recomp.fig', resfn);
    show_res(figname, res, bits_computed_all, 'Bit Recomputations', opts.identifier, opts.override);
    drawnow;
end
end

% -----------------------------------------------------------
function show_res(figname, Y, X, xlb, ttl, override)
try 
    assert(~override);
    openfig(figname);
catch
    [px, py] = avg_curve(Y, X);
    figure, if length(px) == 1, plot(px, py, '+'), else plot(px, py), end
    grid, title(ttl), xlabel(xlb), ylabel('res')
    saveas(gcf, figname);
    if isempty(strfind(computer, 'WIN'))
        unix(['chmod g+w ', figname]); 
        unix(['chmod o-w ', figname]); 
    end
end
end
