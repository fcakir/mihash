function train(trainFunc, run_trial, opts)

global Xtrain Ytrain thr_dist
train_time  = zeros(1, opts.ntrials);
update_time = zeros(1, opts.ntrials);
ht_updates  = zeros(1, opts.ntrials);
bit_flips   = zeros(1, opts.ntrials);
bit_recomp  = zeros(1, opts.ntrials);

num_iters = ceil(opts.noTrainingPoints/opts.batchSize);
myLogInfo('%s: %d train_iters', opts.identifier, num_iters);

if isempty(gcp('nocreate'))
    p = parpool(opts.ntrials);  % get just enough workers
end
parfor t = 1:opts.ntrials
    if run_trial(t) == 0
        myLogInfo('Trial %02d not required, skipped', t);
        continue;
    end
    myLogInfo('%s: random trial %d', opts.identifier, t);
    
    % randomly set test checkpoints (to better mimic real scenarios)
    test_iters      = zeros(1, opts.ntests);
    test_iters(1)   = 1;
    test_iters(end) = num_iters;
    interval = round(num_iters/(opts.ntests-1));
    for i = 1:opts.ntests-2
        iter = interval*i + randi([1 round(interval/3)]) - round(interval/6);
        test_iters(i+1) = iter;
    end
    prefix = sprintf('%s/trial%d', opts.expdir, t);
    
    % train hash functions
    [train_time(t), update_time(t), ht_updates(t), bit_recomp(t), bit_flips(t)] ...
        = trainFunc(Xtrain, Ytrain, thr_dist, prefix, test_iters, t, opts);
end

myLogInfo('Training time (total): %.2f +/- %.2f', mean(train_time), std(train_time));
myLogInfo('HTupdate time (total): %.2f +/- %.2f', mean(update_time), std(update_time));
if strcmp(opts.mapping, 'smooth')
    myLogInfo('    Hash Table Updates (per): %.4g +/- %.4g', mean(ht_updates), std(ht_updates));
    myLogInfo('    Bit Recomputations (per): %.4g +/- %.4g', mean(bit_recomp), std(bit_recomp));
    myLogInfo('    Bit flips (per): %.4g +/- %.4g', mean(bit_flips), std(bit_flips));
end
end
