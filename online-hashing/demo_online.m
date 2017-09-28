function paths = demo_online(method, dataset, nbits, varargin)
% Implementation of an online hashing benchmark as described in: 
%
% "MIHash: Online Hashing with Mutual Information", 
% Fatih Cakir*, Kun He*, Sarah A. Bargal, Stan Sclaroff
% (* equal contribution)
% International Conference on Computer Vision (ICCV) 2017
%
% INPUTS
%   method   - (string) in {'MIHash', 'AdaptHash', 'OKH', 'OSH', 'SketchHash'}
%   dataset  - (string) in {'cifar', 'places', 'labelme'}
%   nbits    - (int) length of binary code, 32 is used in the paper
%   varargin - key-value pairs, see get_opts.m for details
%
% OUTPUTS
%   paths (struct)
%       result - (string) Path to the results file
%       diary  - (string) Path to the experimental log
%       trials - (cell, string) Result files for each trial
%
% A result file is created for each training trial. Such a file has the 
% 'METRIC_trialX.mat' format, and is saved in the results folder specified by 
% opts.dirs.exp. METRIC is the performance metric as indicated by opts.metric,
% and X is the trial number.
%
% A final result file with format 'METRIC_Ntrials.mat' is also created, where 
% N is the total number of trials, equal to opts.ntrials. This file contains the
% average statistics of the individual trial (see test.m).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% parse options
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get method-specific fields first
ip = inputParser;
ip.KeepUnmatched = true;

if strcmp(method, 'MIHash')
    % Implementation of MIHash as described in: 
    %
    % "MIHash: Online Hashing with Mutual Information"
    % Fatih Cakir*, Kun He*, Sarah A. Bargal, Stan Sclaroff (*equal contribution)
    % International Conference on Computer Vision (ICCV) 2017
    %
    % PARAMETERS
    %	no_bins  - (int in [1, nbits]) specifies the number of bins of the 
    %	           histogram (K in Section 3.2)
    % 	stepsize - (float) The learning rate.
    % 	decay    - (float) Decay parameter for learning rate. 
    % 	sigscale - (10) Sigmoid function to smooth the sgn of the hash function,
    % 	           used as second argument to sigmf.m, see Section 3.2
    %     initRS - (int) Initial reservoir size. Must be a positive value. 
    %                    >=500 is recommended. 
    ip.addParamValue('normalize' , true , @islogical);
    ip.addParamValue('no_bins'   , 16   , @isscalar);
    ip.addParamValue('sigscale'  , 10   , @isscalar);
    ip.addParamValue('stepsize'  , 1    , @isscalar);
    ip.addParamValue('decay'     , 0    , @isscalar);
    ip.parse(varargin{:}); opts = ip.Results;

    opts.identifier = sprintf('Bins%dSig%g-S%gD%g', opts.no_bins, ...
        opts.sigscale, opts.stepsize, opts.decay);
    opts.batchSize  = 1;  % hard-coded

elseif strcmp(method, 'AdaptHash')
    % Implementation of AdaptHash as described in: 
    %
    % "Adaptive Hashing for Fast Similarity Search"
    % F. Cakir, S. Sclaroff
    % International Conference on Computer Vision (ICCV) 2015
    %
    % PARAMETERS
    %	alpha 	 - (float) [0, 1] \alpha as in Alg. 1 of AdaptHash. 
    % 	beta 	 - (float) \lambda as in Alg. 1
    % 	stepsize - (float) The learning rate. 
    ip.addParamValue('normalize' , true , @islogical);
    ip.addParamValue('alpha'     , 0.9  , @isscalar);
    ip.addParamValue('beta'      , 1e-2 , @isscalar);
    ip.addParamValue('stepsize'  , 1    , @isscalar);
    ip.parse(varargin{:}); opts = ip.Results;

    opts.identifier = sprintf('A%gB%gS%g', opts.alpha, opts.beta, opts.stepsize);
    opts.batchSize  = 2;  % hard-coded; pair supervision

elseif strcmp(method, 'OKH')
    % Implementation of OKH as described in: 
    %
    % "Online Hashing"
    % L. K. Huang, Q. Y. Yang and W. S. Zheng
    % International Joint Conference on Artificial Intelligence (IJCAI) 2013
    %
    % PARAMETERS
    %	c 	 - (float) Parameter C as in Alg. 1 of OKH. 
    % 	alpha	 - (float) \alpha as in Eq. 3 of OKH
    ip.addParamValue('normalize' , true , @islogical);
    ip.addParamValue('c'         , 0.1  , @isscalar);
    ip.addParamValue('alpha'     , 0.2  , @isscalar);
    ip.parse(varargin{:}); opts = ip.Results;

    opts.identifier = sprintf('C%gA%g', opts.c, opts.alpha);
    opts.batchSize  = 2;  % hard-coded; pair supervision

elseif strcmp(method, 'OSH')
    % Implementation of the OSH method as described in: 
    %
    % "Online Supervised Hashing"
    % F. Cakir, S. Sclaroff
    % International Conference on Image Processing (ICIP) 2015
    %
    % "Online Supervised Hashing"
    % F. Cakir, S. A. Bargal, S. Sclaroff
    % Computer Vision and Image Understanding (CVIU) 2016
    %
    % PARAMETERS
    % 	stepsize - (float) The learning rate.
    % 	SGDBoost - (int)   Choices are {0, 1}.  SGDBoost=1 corresponds to do the 
    % 			   online boosting formulation with exponential loss as 
    % 			   described in the above papers. SGDBoost=0, corresponds
    % 			   to a hinge loss formulation without the online boosting 
    % 			   approach. SGDBoost=0 typically works better.
    ip.addParamValue('normalize' , true , @islogical);
    ip.addParamValue('stepsize'  , 0.1  , @isscalar);
    ip.addParamValue('SGDBoost'  , 1    , @isscalar);
    ip.parse(varargin{:}); opts = ip.Results;

    opts.identifier = sprintf('B%dS%g', opts.SGDBoost, opts.stepsize);
    opts.batchSize  = 1;      % hard-coded

elseif strcmp(method, 'SketchHash')
    % Implementation of the SketchHash method as described in: 
    %
    % "Online Sketching Hashing"
    % C. Leng, J. Wu, J. Cheng, X. Bai and H. Lu
    % Computer Vision and Pattern Recognition (CVPR) 2015
    %
    % PARAMETERS
    % 	sketchSize - (int) size of the sketch matrix.
    % 	 batchSize - (int) The batch size, i.e. size of the data chunk
    ip.addParamValue('normalize'  , false , @islogical);
    ip.addParamValue('sketchSize' , 200   , @isscalar);
    ip.addParamValue('batchSize'  , 50    , @isscalar);
    ip.parse(varargin{:}); opts = ip.Results;

    opts.identifier = sprintf('Ske%dBat%d', opts.sketchSize, opts.batchSize);
    assert(opts.batchSize>=nbits, 'Sketching needs batchSize>=nbits');

else
    error('Implemented methods: {MIHash, AdaptHash, OKH, OSH, SketchHash}');
end
opts.methodID = method;

% get generic fields + necessary preparation
opts = opts_online(opts, dataset, nbits, varargin{:});


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% online hashing demo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if opts.override  % will purge expdir, use with care
    logInfo('OVERRIDE: deleting existing results!');
    unix(['rm -rfv ', fullfile(opts.dirs.exp, '*')]);
end

% ---------------------------------------------------------------------
% 1. Set up paths
% ---------------------------------------------------------------------
prefix = fullfile(opts.dirs.exp, opts.metric);
paths  = [];
paths.result = sprintf('%s_%dtrials.mat', prefix, opts.ntrials);
paths.trials = arrayfun(@(t) sprintf('%s/trial%d.mat', opts.dirs.exp, t), ...
    1:opts.ntrials, 'uniform', false);
res_exist    = cellfun(@(f) exist(f, 'file'), paths.trials);
paths.diary  = record_diary(opts.dirs.exp, ~all(res_exist));


% ---------------------------------------------------------------------
% 2. load data (only if necessary)
% ---------------------------------------------------------------------
global Dataset
if isfield(Dataset, 'name') && strcmp(Dataset.name, opts.dataset)
    logInfo('Dataset [%s] already loaded', Dataset.name);
elseif ~all(res_exist)
    datasetFunc  = str2func(['Datasets.' opts.dataset]);
    Dataset      = datasetFunc(opts);
    Dataset.name = opts.dataset;
end
% thr_dist may be used in computing affinity matrices
opts.thr_dist = Dataset.thr_dist;


% ---------------------------------------------------------------------
% 3. TRAINING
% ---------------------------------------------------------------------
logInfo('[%s] %s: Training ...', opts.methodID, opts.identifier);
methodFunc = str2func(['Methods.' method]);
methodObj  = methodFunc();

% NOTE: if you have the Parallel Computing Toolbox, you can use parfor 
%       to run the trials in parallel
info_all = [];
for t = 1:opts.ntrials
    logInfo('[%s] %s: random trial %d', opts.methodID, opts.identifier, t);
    rng(opts.randseed+t, 'twister'); % fix randseed for reproducible results

    if res_exist(t)
        info = load(paths.trials{t});
        logInfo('Results exist');
    else
        % randomly set test checkpoints
        num_iters  = ceil(opts.numTrain*opts.epoch/opts.batchSize);
        test_iters = zeros(1, opts.ntests-2);
        interval   = round(num_iters/(opts.ntests-1));
        for i = 1:opts.ntests-2
            iter = interval*i + randi([1 round(interval/3)]) - round(interval/6);
            test_iters(i) = iter;
        end
        test_iters = [1, test_iters, num_iters];  % always include 1st & last
        
        % train hashing method
        info = train_online(methodObj, Dataset, t, paths.trials{t}, test_iters, opts);
    end
    info_all = [info_all, info];
end
logInfo('[%s] %s: Training is done.', opts.methodID, opts.identifier);

reportStat = @(field, str, fmt) fprintf(['%s: ' fmt ' +/- ' fmt '\n'], str, ...
    mean(arrayfun(@(x) x.(field)(end), info_all)), ...
    std (arrayfun(@(x) x.(field)(end), info_all)));

fprintf('================================================\n');
reportStat('time_train' , '       Training Time', '%.2f');
reportStat('time_update', '      HT update time', '%.2f');
reportStat('time_reserv', '      Reservoir time', '%.2f');
fprintf('\n');
reportStat('ht_updates' , '  Hash Table Updates', '%.4g');
reportStat('bit_recomp' , '  Bit Recomputations', '%.3d');
fprintf('================================================\n');


% ---------------------------------------------------------------------
% 4. TESTING: see test_online
% ---------------------------------------------------------------------
logInfo('[%s] %s: Testing ...', opts.methodID, opts.identifier);
try
    info_all = load(paths.result);
catch
    info_all = [];
    for t = 1:opts.ntrials
        info = test_online(Dataset, t, opts);
        info_all = [info_all, info];
    end
    save(paths.result, 'info_all');
end
auc   = arrayfun(@(x) mean(x.metric), info_all);
final = arrayfun(@(x) x.metric(end) , info_all);
logInfo('');
logInfo('  AUC %s: %.3g +/- %.3g', opts.metric, mean(auc), std(auc));
logInfo('FINAL %s: %.3g +/- %.3g', opts.metric, mean(final), std(final));


% ---------------------------------------------------------------------
% 5. Done
% ---------------------------------------------------------------------
logInfo('All done.');
logInfo('Results file: %s', paths.result);
logInfo('  Diary file: %s', paths.diary);
diary('off');

end
