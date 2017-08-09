function train(trainFunc, run_trial, opts)
% This is the routine which calls different training subroutines based on the 
% hashing method. Separate trials are executed here and rudimentary statistics 
% are computed and displayed. 
%
% INPUTS
%    trainFunc - (func handle) Function handle determining which training routine
% 			       to call
%    run_trial - (vector)      Boolean vector specifying which trials to run.
% 			       if opts.override=0, previously ran trials are skipped.
%	opts   - (struct)      Parameter structure.
% OUTPUTS
% 	none

global Xtrain Ytrain thr_dist

% time to learn the hash mapping
train_time  = zeros(1, opts.ntrials);
% time to update the hash table
update_time = zeros(1, opts.ntrials);
% time to update/maintain the reservoir
reservoir_time = zeros(1, opts.ntrials);
% number of hash table updates performed
ht_updates  = zeros(1, opts.ntrials);
% number of bit flips occured in the hash table, see update_hash_table.m
bit_flips   = zeros(1, opts.ntrials);
% number of bit recomputations, generally this equal ht_updates x hashcode length
% x number of hashed items, see update_hash_table.m 
bit_recomp  = zeros(1, opts.ntrials);

num_iters = ceil(opts.noTrainingPoints*opts.epoch/opts.batchSize);
logInfo('%s: %d train_iters', opts.identifier, num_iters);

% NOTE: you can use parfor to run trials in parallel
for t = 1:opts.ntrials
    rng(opts.randseed+t, 'twister'); % fix randseed for reproducible results
    if ~run_trial(t)
        logInfo('Trial %02d not required, skipped', t);
        continue;
    end
    logInfo('%s: random trial %d', opts.identifier, t);
    
    % randomly set test checkpoints
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
    [train_time(t), update_time(t), resservoir_time(t), ...
        ht_updates(t), bit_recomp(t), bit_flips(t)] = ...
        trainFunc(Xtrain, Ytrain, thr_dist, prefix, test_iters, t, opts);
end

logInfo(' Training time (total): %.2f +/- %.2f', mean(train_time), std(train_time));
logInfo('HT_update time (total): %.2f +/- %.2f', mean(update_time), std(update_time));
logInfo('Reservoir time (total): %.2f +/- %.2f', mean(resservoir_time), std(resservoir_time));
logInfo('');
logInfo('Hash Table Updates (per): %.4g +/- %.4g', mean(ht_updates), std(ht_updates));
logInfo('Bit Recomputations (per): %.4g +/- %.4g', mean(bit_recomp), std(bit_recomp));
logInfo('         Bit flips (per): %.4g +/- %.4g', mean(bit_flips), std(bit_flips));
end
