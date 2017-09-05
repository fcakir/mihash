function info = test_online(Dataset, trial, opts)
% Computes the performance by loading and evaluating the "checkpoint" files
% saved during training. 
%
% INPUTS
%       paths (struct)
%               result - (string) final result path
%               diary  - (string) exp log path
%               trials - (cell, string) paths of result for each trial
%	opts  (struct) 
% 
% OUTPUTS
%	none


Xtrain = Dataset.Xtrain;
Ytrain = Dataset.Ytrain;
Xtest  = Dataset.Xtest;
Ytest  = Dataset.Ytest;
Aff    = affinity(Xtrain, Xtest, Ytrain, Ytest, opts);

prefix = sprintf('%s/trial%d', opts.expdir, trial);
model  = load([prefix '.mat']);

% handle transformations to X
if strcmp(opts.methodID, 'okh')
    % do kernel mapping for test data
    testX_t = exp(-0.5*sqdist(Xtest', model.Xanchor')/model.sigma^2)';
    testX_t = [testX_t; ones(1,size(testX_t,2))]';
elseif strcmp(opts.methodID, 'sketch')
    % subtract mean
    testX_t = bsxfun(@minus, Xtest, model.instFeatAvePre);
else
    % OSH, AdaptHash: nothing
    testX_t = Xtest;
end

info = struct(...
    'metric'         , [] , ...
    'train_time'     , [] , ...
    'train_iter'     , [] , ...
    'train_examples' , [] , ...
    'bits_computed'  , [] );

for i = 1:length(model.test_iters)
    iter = model.test_iters(i);
    fprintf('Trial %d, Checkpoint %5d/%d, ', trial, iter*opts.batchSize, ...
        opts.numTrain*opts.epoch);

    % determine whether to actually run test or not
    % if there's no HT update since last test, just copy results
    if i == 1
        runtest = true;
    else
        st = model.test_iters(i-1);
        ed = model.test_iters(i);
        runtest = any(model.update_iters>st & model.update_iters<=ed);
    end

    d = load(sprintf('%s_iter/%d.mat', prefix, iter));
    if runtest
        % NOTE: for intermediate iters, need to use W_snapshot (not W!)
        %       to compute Htest, to make sure it's computed using the same
        %       hash mapping as Htrain.
        Htest  = (testX_t * d.W_snapshot > 0)';
        Htrain = d.H;
        metric(i) = evaluate(Htrain, Htest, Ytrain, Ytest, opts, Aff);
        info.bits_computed(i) = d.bits_computed;
    else
        info.metric(i) = info.metric(i-1);
        info.bits_computed(i) = info.bits_computed(i-1);
        fprintf(' %g\n', info.metric(i));
    end
    info.train_time(i) = d.train_time;
    info.train_iter(i) = iter;
    info.train_examples(i) = iter * opts.batchSize;
end

end
