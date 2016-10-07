function test_osh(resfn, res_trial_fn, res_exist, opts)
% if we're running this function, it means some elements in res_exist is false
% and we need to compute/recompute the corresponding res_trial_fn's
global Xtest Ytest Ytrain

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
else
    cateTrainTest = (trainY * testY' > 0);
end

clear res bitflips train_iter train_time
for t = 1:opts.ntrials
    if res_exist(t)
        myLogInfo('Trial %d: results exist', t);
        load(res_trial_fn{t});
    else
        clear t_res t_bits_computed_all  t_bitflips t_train_iter t_train_time
        Tprefix = sprintf('%s/trial%d', opts.expdir, t);
        trial_model = load(sprintf('%s.mat', Tprefix));
        for i = 1:length(trial_model.test_iters)  % may NOT be 1:opts.ntests!
            iter = trial_model.test_iters(i);
            d = load(sprintf('%s_iter%d.mat', Tprefix, iter));
            Htrain = d.H;

            % AdaptHash uses test_osh, currently it doesn't work with the 
            % 'label arriving strategy' scenario. 
            if strcmp(caller,'demo_adapthash') 
                ind = 1:size(testX,1);
            else
                % We're removing test items in which their labels have
                % not been observed. However this can cause huge
                % flunctuations in performance at the beginning. For
                % instance at very first iteration we've seen only a
                % label, thus we remove all items that do not belong to that
                % label from the test. Depending on the label, the mAP
                % can be very high or low. This should depend on the
                % testing scenario, imo, for default and 'smooth'
                % mapping simply apply the hash mapping to all test and
                % train data -and report the performance.
                if opts.tstScenario == 2
                    ind = ismember(testY,unique(d.seenLabels));
                else
                    ind = 1:size(testX, 1);
                end
            end

            Htest  = (d.W'*testX(ind,:)' > 0);

            fprintf('Trial %d, Ex %5d/%d, ', t, iter*opts.batchSize, opts.noTrainingPoints);

            %		ok = opts.prec_k;
            %		if size(Htrain,2) < opts.prec_k	
            %			opts.prec_k = size(Htrain,2);
            %		end
            t_res(i) = get_results(Htrain, Htest, trainY(1:size(Htrain,2)), testY(ind), opts, cateTrainTest);
            %		opts.prec_k = ok;
            t_bits_computed_all(i) = d.bits_computed_all;
            t_bitflips(i) = d.bitflips;
            t_train_iter(i) = iter;
            t_train_time(i) = d.train_time;
        end
        clear Htrain Htest Ltest
        save(res_trial_fn{t}, 't_res', 't_bitflips', 't_train_iter', 't_train_time','t_bits_computed_all');
    end
    res(t, :) = t_res;
    bitflips(t, :) = t_bitflips;
    bits_computed_all(t, :) = t_bits_computed_all;
    train_iter(t, :) = t_train_iter;
    train_time(t, :) = t_train_time;
end
myLogInfo('Final test %s: %.3g +/- %.3g', ...
    opts.metric, mean(res(:,end)), std(res(:,end)));

% save all trials in a single file (for backward compatibility)
% it may overwrite existing file, but whatever
save(resfn, 'res', 'bitflips', 'train_iter', 'train_time','bits_computed_all');

% visualize
if opts.showplots
    % draw curves, with auto figure saving
    figname = sprintf('%s_iter.fig', resfn);
    show_res(figname, res, train_iter, 'Training Examples', opts.identifier, opts.override);
    figname = sprintf('%s_cpu.fig', resfn);
    show_res(figname, res, train_time, 'CPU Time', opts.identifier, opts.override);
    figname = sprintf('%s_flip.fig', resfn);
    show_res(figname, res, bitflips, 'Bit Flips', opts.identifier, opts.override);
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
end
end
