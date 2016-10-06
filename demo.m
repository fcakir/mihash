function resfn = demo(opts, trainFunc, testFunc)
% PARAMS
%  ftype (string) from {'gist', 'cnn'}
%  dataset (string) from {'cifar', 'sun','nus'}
%  nbits (integer) is length of binary code
%  varargin: see get_opts.m for details
%  

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
            run_trial(t) = 0; continue;
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
global Xtrain Xtest Ytrain Ytest Dtype
Dtype_this = [dataset '_' ftype];
if ~isempty(Dtype) && strcmp(Dtype_this, Dtype)
    myLogInfo('Dataset already loaded for %s', Dtype_this);
elseif (any(run_trial) || ~all(res_exist))
    myLogInfo('Loading data for %s...', Dtype_this);
    eval(['[Xtrain, Ytrain, Xtest, Ytest] = load_' opts.ftype '(dataset, opts);']);
    Dtype = Dtype_this;
end


% 3. TRAINING: run all _necessary_ trials (handled by train_osh)
if any(run_trial)
    myLogInfo('Training models...');
    trainFunc(run_trial, opts);
end
myLogInfo('Training is done.');


% 4. TESTING: run all _necessary_ trials
if ~all(res_exist) || ~exist(resfn, 'file')
    myLogInfo('Testing models...');
    testFunc(resfn, res_trial_fn, res_exist, opts);
end
myLogInfo('Testing is done.');

end
