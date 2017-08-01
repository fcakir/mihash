function [resfn, diary_path] = demo(opts, trainFunc)
% Creates the experimental results files and determines whether it is necessary 
% to do training and testing. 
% A result file is created for each trial (see opts.ntrials in get_opts.m)
% parameter. Such a file has the METRIC_trialX.mat format and is saved in the 
% results folder specified by opts.expdir. METRIC is the performance metric as 
% indicated by opts.metric and X is the trial number. This file contains the 
% performance values at each "checkpoint" among other information.
% A final result file having format METRIC_Ntrials.mat is created where N is the
% total number of trials, equal to opts.ntrials. This file contains the average
% statistics of the individual trial (see test.m).
%
% INPUTS
%	opts   - (struct)      Parameter structure.
%    trainFunc - (func handle) Function handle determining which training routine
% 			       to call
%    testFunc  - (func handle) Function handle determining which test routine to 
% 			       call. Note in the current version there is only 
% 			       a single test routine for all methods. 
% OUTPUTS
% 	resfn - (string)       Path to final results file. 
%  diary_path - (string)       Path to the diary containing the command window
%			       output.

% 0. result files
Rprefix = sprintf('%s/%s', opts.expdir, opts.metric);
if opts.testFrac < 1
    Rprefix = sprintf('%s_frac%g', Rprefix);
end
resfn = sprintf('%s_%dtrials.mat', Rprefix, opts.ntrials);
res_trial_fn = cell(1, opts.ntrials);
for t = 1:opts.ntrials 
    res_trial_fn{t} = sprintf('%s_trial%d.mat', Rprefix, t);
end
if opts.override
    res_exist = zeros(1, opts.ntrials);
else
    res_exist = cellfun(@(r) exist(r, 'file'), res_trial_fn);
end


% 1. determine which (training) trials to run
if opts.override
    run_trial = ones(1, opts.ntrials);
else
    run_trial = zeros(1, opts.ntrials);
    for t = 1:opts.ntrials
        if exist(res_trial_fn{t}, 'file')
            run_trial(t) = 0; 
            continue;
        end
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


% 2. load data (only if necessary)
global Xtrain Xtest Ytrain Ytest thr_dist Dtype
Dtype_this = [opts.dataset '_' opts.ftype];
if ~isempty(Dtype) && strcmp(Dtype_this, Dtype)
    logInfo('Dataset already loaded for %s', Dtype_this);
elseif (any(run_trial) || ~all(res_exist))
    logInfo('Loading data for %s...', Dtype_this);
    if strcmp(opts.methodID, 'sketch')
        eval(['[Xtrain, Ytrain, Xtest, Ytest, thr_dist] = load_' opts.ftype '(opts, false);']);
    else
        eval(['[Xtrain, Ytrain, Xtest, Ytest, thr_dist] = load_' opts.ftype '(opts);']);
    end
    Dtype = Dtype_this;
end


% 3. TRAINING: run all _necessary_ trials (handled by train.m)
if any(run_trial)
    logInfo('Training models...');
    train(trainFunc, run_trial, opts);
end
logInfo('%s: Training is done.', opts.identifier);


% 4. TESTING: run all _necessary_ trials
if ~all(res_exist) || ~exist(resfn, 'file')
    logInfo('Testing models...');
    test(resfn, res_trial_fn, res_exist, opts);
end
logInfo('%s: Testing is done.', opts.identifier);

diary_path = opts.diary_name;

% 5. close parpool, if any
%set_parpool(0);
end
