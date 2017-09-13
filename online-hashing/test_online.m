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


Aff = affinity(Dataset.Xtrain, Dataset.Xtest, Dataset.Ytrain, Dataset.Ytest, opts);

prefix = sprintf('%s/trial%d', opts.dirs.exp, trial);
model  = load([prefix '.mat']);

% handle transformations to X
if strcmp(opts.methodID, 'okh')
elseif strcmp(opts.methodID, 'sketch')
    % subtract mean
    testX_t = bsxfun(@minus, Dataset.Xtest, model.instFeatAvePre);
else
    % OSH, AdaptHash: nothing
    testX_t = Dataset.Xtest;
end

info = struct(...
    'metric'         , [] , ...
    'train_time'     , [] , ...
    'train_iter'     , [] , ...
    'train_examples' , [] , ...
    'bit_recomp'     , [] );

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
        %Htest  = (testX_t * d.W_snapshot) > 0;
        Htest  = methodObj.encode(d.W_snapshot, Dataset.Xtest, true);
        Htrain = d.H;
        info.metric(i) = evaluate(Htrain, Htest, opts, Aff);
        info.bit_recomp(i) = d.bit_recomp;
    else
        info.metric(i) = info.metric(i-1);
        info.bit_recomp(i) = info.bit_recomp(i-1);
        fprintf(' %g\n', info.metric(i));
    end
    info.train_time(i) = d.time_train;
    info.train_iter(i) = iter;
    info.train_examples(i) = iter * opts.batchSize;
end

end
