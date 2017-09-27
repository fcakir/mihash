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

info = struct(...
    'metric'         , [] , ...
    'train_time'     , [] , ...
    'train_iter'     , [] , ...
    'train_examples' , [] , ...
    'bit_recomp'     , [] );

testX = Dataset.Xtest;
Aff = affinity(Dataset.Xtrain, Dataset.Xtest, Dataset.Ytrain, Dataset.Ytest, opts);

prefix = sprintf('%s/trial%d', opts.dirs.exp, trial);
model = load([prefix '.mat']);

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

    itmd = load(sprintf('%s_iter/%d.mat', prefix, iter));
    P = itmd.params;
    if strcmp(opts.methodID, 'OKH')
        % do kernel mapping for test data
        testX = exp(-0.5*sqdist(Dataset.Xtest', P.Xanchor')/P.sigma^2)';
        testX = [testX; ones(1,size(testX,2))]';
    elseif strcmp(opts.methodID, 'SketchHash')
        % subtract estimated mean
        testX = bsxfun(@minus, Dataset.Xtest, P.instFeatAvePre);
    end
    if runtest
        % NOTE: for intermediate iters, need to use Wsnapshot (not W!)
        %       to compute Htest, to make sure it's computed using the same
        %       hash mapping as Htrain.
        Htest  = (testX * itmd.Wsnapshot) > 0;
        Htrain = itmd.H;
        info.metric(i) = evaluate(Htrain, Htest, opts, Aff);
        info.bit_recomp(i) = itmd.bit_recomp;
    else
        info.metric(i) = info.metric(i-1);
        info.bit_recomp(i) = info.bit_recomp(i-1);
        fprintf(' %g\n', info.metric(i));
    end
    info.train_time(i) = itmd.time_train;
    info.train_iter(i) = iter;
    info.train_examples(i) = iter * opts.batchSize;
end

end
