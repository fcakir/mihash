function [resPath, diaPath] = demo_online(method, ftype, dataset, nbits, varargin)
% Implementation of AdaptHash as described in: 
%
% Fatih Cakir*, Kun He*, Sarah Adel Bargal, Stan Sclaroff 
% "MIHash: Online Hashing with Mutual Information", (*equal contribution).
% International Conference on Computer Vision (ICCV) 2015
%
% INPUTS
%   method   - (string) from {'mihash', 'adapt', 'okh', 'osh', 'sketch'}
%   ftype    - (string) from {'gist', 'cnn'}
%   dataset  - (string) from {'cifar', 'sun','nus'}
%   nbits    - (integer) is length of binary code
%   varargin - see get_opts.m for details
% OUTPUTS
%   resPath - (string) Path to the results file. see demo.m .
%   diaPath - (string) Path to the diary which contains the command window text


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get opts
%
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

    opts.identifier = sprintf('B%d_S%g', opts.SGDBoost, opts.stepsize);
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
% run demo
%
% 0. result files
Rprefix = sprintf('%s/%s', opts.expdir, opts.metric);
if opts.testFrac < 1
    Rprefix = sprintf('%s_frac%g', Rprefix);
end
resPath = sprintf('%s_%dtrials.mat', Rprefix, opts.ntrials);
resPathT = cell(1, opts.ntrials);
for t = 1:opts.ntrials 
    resPathT{t} = sprintf('%s_trial%d.mat', Rprefix, t);
end
if opts.override
    res_exist = zeros(1, opts.ntrials);
else
    res_exist = cellfun(@(r) exist(r, 'file'), resPathT);
end


% 1. determine which (training) trials to run
if opts.override
    run_trial = ones(1, opts.ntrials, 'logical');
else
    run_trial = cellfun(@(f) exist(f, 'file'), resPathT);
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
if ~all(res_exist) || ~exist(resPath, 'file')
    logInfo('Testing models...');
    test(resPath, resPathT, res_exist, opts);
end
logInfo('%s: Testing is done.', opts.identifier);

diaPath = opts.diary_name;
diary('off');
end
