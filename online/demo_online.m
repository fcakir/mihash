function [res_path, dia_path] = demo_online(method, ftype, dataset, nbits, varargin)
% Implementation of an online hashing benchmark as described in: 
%
% Fatih Cakir*, Kun He*, Sarah Adel Bargal, Stan Sclaroff
% (* equal contribution)
% "MIHash: Online Hashing with Mutual Information", 
% International Conference on Computer Vision (ICCV) 2017
%
% INPUTS
%   method   - (string) from {'mihash', 'adapt', 'okh', 'osh', 'sketch'}
%   ftype    - (string) feature type, from {'gist', 'cnn'}
%   dataset  - (string) from {'cifar', 'sun','nus'}
%   nbits    - (integer) length of binary code, 32 is used in the paper
%   varargin - key-value pairs, see get_opts.m for details
% OUTPUTS
%   res_path - (string) Path to the results file
%   dia_path - (string) Path to the experimental log


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get opts

% get method-specific fields first
ip = inputParser;
ip.KeepUnmatched = true;

if strcmp(method, 'mihash')
    ip.addParamValue('no_bins', 16, @isscalar);
    ip.addParamValue('stepsize', 1, @isscalar);
    ip.addParamValue('decay', 0, @isscalar);
    ip.addParamValue('sigscale', 10, @isscalar);
    ip.addParamValue('initRS', 500, @isscalar); % initial size of reservoir
    ip.parse(varargin{:}); opts = ip.Results;

    opts.identifier = sprintf('Bins%dSig%g_Step%gDecay%g_InitRS%g', opts.no_bins, ...
        opts.sigscale, opts.stepsize, opts.decay, opts.initRS);
    opts.batchSize  = 1;  % hard-coded

elseif strcmp(method, 'adapt')
    ip.addParamValue('alpha', 0.9, @isscalar);
    ip.addParamValue('beta', 1e-2, @isscalar);
    ip.addParamValue('stepsize', 1, @isscalar);
    ip.parse(varargin{:}); opts = ip.Results;

    opts.identifier = sprintf('A%gB%gS%g', opts.alpha, opts.beta, opts.stepsize);
    opts.batchSize  = 2;  % hard-coded; pair supervision

elseif strcmp(method, 'okh')
    ip.addParamValue('c', 0.1, @isscalar);
    ip.addParamValue('alpha', 0.2, @isscalar);
    ip.parse(varargin{:}); opts = ip.Results;

    opts.identifier = sprintf('C%gA%g', opts.c, opts.alpha);
    opts.batchSize  = 2;  % hard-coded; pair supervision

elseif strcmp(method, 'osh')
    ip.addParamValue('stepsize', 0.1, @isscalar);
    ip.addParamValue('SGDBoost', 1, @isscalar);
    ip.parse(varargin{:}); opts = ip.Results;

    opts.identifier = sprintf('B%dS%g', opts.SGDBoost, opts.stepsize);
    opts.batchSize  = 1;      % hard-coded

elseif strcmp(method, 'sketch')
    ip.addParamValue('sketchSize', 200, @isscalar);
    ip.addParamValue('batchSize', 50, @isscalar);
    ip.parse(varargin{:}); opts = ip.Results;

    opts.identifier = sprintf('Ske%dBat%d', opts.sketchSize, opts.batchSize);
    assert(opts.batchSize>=nbits, 'Sketching needs batchSize>=nbits');

else
    error('Implemented methods: {mihash, adapt, okh, osh, sketch}');
end
opts.methodID = method;

% get generic fields + necessary preparation
opts = get_opts(opts, ftype, dataset, nbits, varargin{:});


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% run online hashing demo

% 1. determine which trials to run
prefix = fullfile(opts.expdir, opts.metric);
res_path = sprintf('%s_%dtrials.mat', prefix, opts.ntrials);
trial_path = arrayfun(@(t) sprintf('%s_trial%d.mat', prefix, t), 1:opts.ntrials, ...
    'uniform', false);
if opts.override
    unix(['rm -fv ', fullfile(opts.expdir, 'diary*')]);
    run_trial = ones(1, opts.ntrials, 'logical');
else
    run_trial = cellfun(@(f) ~exist(f, 'file'), trial_path);
end


% 2. load data (only if necessary)
global Xtrain Xtest Ytrain Ytest thr_dist Dtype
Dtype1 = [opts.dataset '_' opts.ftype];
if ~isempty(Dtype) && strcmp(Dtype1, Dtype)
    logInfo('Dataset already loaded for %s', Dtype);
elseif any(run_trial)
    Dtype = Dtype1;
    logInfo('Loading data for %s...', Dtype);
    featureFunc = str2func(['load_' opts.ftype]);
    [Xtrain, Ytrain, Xtest, Ytest, thr_dist] = ...
        featureFunc(opts, ~strcmp(opts.methodID, 'sketch'));
end


% hold a diary -save it to opts.expdir
diaryName = @(x) sprintf('%s/diary_%03d.txt', opts.expdir, x);
index = 1;
while exist(diaryName(index), 'file'), index = index + 1; end
dia_path = diaryName(index);

if any(run_trial)
    diary(dia_path); diary('on');

    % 3. TRAINING: run all _necessary_ trials
    logInfo('Training models...');
    trainFunc = str2func(['train_' method]);
    train(trainFunc, run_trial, opts);
    logInfo('%s: Training is done.', opts.identifier);

    % 4. TESTING: run all _necessary_ trials
    logInfo('Testing models...');
    test(res_path, trial_path, run_trial, opts);
    logInfo('%s: Testing is done.', opts.identifier);
end
    
% 5. Done
logInfo('All done.');
logInfo('Results file: %s', res_path);
logInfo('  Diary file: %s', dia_path);
diary('off');

end
