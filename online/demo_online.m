function [res_path, dia_path] = demo_online(method, ftype, dataset, nbits, varargin)
% Copyright (c) 2017, Fatih Cakir, Kun He, Saral Adel Bargal, Stan Sclaroff 
% All rights reserved.
% 
% If used for please cite the below paper:
%
% "MIHash: Online Hashing with Mutual Information", 
% Fatih Cakir*, Kun He*, Sarah Adel Bargal, Stan Sclaroff
% (* equal contribution)
% International Conference on Computer Vision (ICCV) 2017
% 
% Usage of code from authors not listed above might be subject
% to different licensing. Please check with the corresponding authors (see below)
% for additional information.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
% 
% 1. Redistributions of source code must retain the above copyright notice, this
%    list of conditions and the following disclaimer.
% 2. Redistributions in binary form must reproduce the above copyright notice,
%    this list of conditions and the following disclaimer in the documentation
%    and/or other materials provided with the distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
% ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
% ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
% ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
% 
% The views and conclusions contained in the software and documentation are those
% of the authors and should not be interpreted as representing official policies,
% either expressed or implied, of the FreeBSD Project.
%
%------------------------------------------------------------------------------
% INPUTS
%   method   - (string) from {'mihash', 'adapt', 'okh', 'osh', 'sketch'}
%   ftype    - (string) feature type, from {'gist', 'cnn'}
%   dataset  - (string) from {'cifar', 'sun','nus'}
%   nbits    - (integer) length of binary code, 32 is used in the paper
%   varargin - key-value pairs, see get_opts.m for details
%
% OUTPUTS
%   res_path - (string) Path to the results file
%   dia_path - (string) Path to the experimental log
%
% A result file is created for each training trial. Such a file has the 
% 'METRIC_trialX.mat' format, and is saved in the results folder specified by 
% opts.expdir. METRIC is the performance metric as indicated by opts.metric,
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
    % Implementation of MIHash. If used please cite below paper: 
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
    ip.addParamValue('normalize', true, @islogical);
    ip.addParamValue('no_bins', 16, @isscalar);
    ip.addParamValue('stepsize', 1, @isscalar);
    ip.addParamValue('decay', 0, @isscalar);
    ip.addParamValue('sigscale', 10, @isscalar);
    ip.addParamValue('initRS', 500, @isscalar); % initial size of reservoir
    ip.parse(varargin{:}); opts = ip.Results;

    opts.identifier = sprintf('Bins%dSig%g_Step%gDecay%g_InitRS%g', opts.no_bins, ...
        opts.sigscale, opts.stepsize, opts.decay, opts.initRS);
    opts.batchSize  = 1;  % hard-coded

elseif strcmp(method, 'AdaptHash')
    % Implementation of AdaptHash. If used please cite below paper:
    %
    % "Adaptive Hashing for Fast Similarity Search"
    % F. Cakir, S. Sclaroff
    % International Conference on Computer Vision (ICCV) 2015
    %
    % PARAMETERS
    %	alpha 	 - (float) [0, 1] \alpha as in Alg. 1 of AdaptHash. 
    % 	beta 	 - (float) \lambda as in Alg. 1
    % 	stepsize - (float) The learning rate. 
    ip.addParamValue('normalize', true, @islogical);
    ip.addParamValue('alpha', 0.9, @isscalar);
    ip.addParamValue('beta', 1e-2, @isscalar);
    ip.addParamValue('stepsize', 1, @isscalar);
    ip.parse(varargin{:}); opts = ip.Results;

    opts.identifier = sprintf('A%gB%gS%g', opts.alpha, opts.beta, opts.stepsize);
    opts.batchSize  = 2;  % hard-coded; pair supervision

elseif strcmp(method, 'OKH')
    % Implementation of OKH. If used please cite below paper:
    %
    % "Online Hashing"
    % L. K. Huang, Q. Y. Yang and W. S. Zheng
    % International Joint Conference on Artificial Intelligence (IJCAI) 2013
    %
    % PARAMETERS
    %	c 	 - (float) Parameter C as in Alg. 1 of OKH. 
    % 	alpha	 - (float) \alpha as in Eq. 3 of OKH
    ip.addParamValue('normalize', true, @islogical);
    ip.addParamValue('c', 0.1, @isscalar);
    ip.addParamValue('alpha', 0.2, @isscalar);
    ip.parse(varargin{:}); opts = ip.Results;

    opts.identifier = sprintf('C%gA%g', opts.c, opts.alpha);
    opts.batchSize  = 2;  % hard-coded; pair supervision

elseif strcmp(method, 'OSH')
    % Implementation of the OSH. If used please cite below paper: 
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
    ip.addParamValue('normalize', true, @islogical);
    ip.addParamValue('stepsize', 0.1, @isscalar);
    ip.addParamValue('SGDBoost', 1, @isscalar);
    ip.parse(varargin{:}); opts = ip.Results;

    opts.identifier = sprintf('B%dS%g', opts.SGDBoost, opts.stepsize);
    opts.batchSize  = 1;      % hard-coded

elseif strcmp(method, 'SketchHash')
    % Implementation of the SketchHash. If used please cite below paper:
    %
    % "Online Sketching Hashing"
    % C. Leng, J. Wu, J. Cheng, X. Bai and H. Lu
    % Computer Vision and Pattern Recognition (CVPR) 2015
    %
    % PARAMETERS
    % 	sketchSize - (int) size of the sketch matrix.
    % 	 batchSize - (int) The batch size, i.e. size of the data chunk
    ip.addParamValue('normalize', false, @islogical);
    ip.addParamValue('sketchSize', 200, @isscalar);
    ip.addParamValue('batchSize', 50, @isscalar);
    ip.parse(varargin{:}); opts = ip.Results;

    opts.identifier = sprintf('Ske%dBat%d', opts.sketchSize, opts.batchSize);
    assert(opts.batchSize>=nbits, 'Sketching needs batchSize>=nbits');

else
    error('Implemented methods: {MIHash, AdaptHash, OKH, OSH, SketchHash}');
end
opts.methodID = method;

% get generic fields + necessary preparation
opts = get_opts(opts, ftype, dataset, nbits, varargin{:});


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% online hashing demo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% ---------------------------------------------------------------------
% 1. determine which trials to run
% ---------------------------------------------------------------------
prefix = fullfile(opts.expdir, opts.metric);
res_path = sprintf('%s_%dtrials.mat', prefix, opts.ntrials);
trial_path = arrayfun(@(t) sprintf('%s_trial%d.mat', prefix, t), ...
    1:opts.ntrials, 'uniform', false);
if opts.override
    unix(['rm -fv ', fullfile(opts.expdir, 'diary*')]);
    run_trial = ones(1, opts.ntrials, 'logical');
else
    run_trial = cellfun(@(f) ~exist(f, 'file'), trial_path);
end
if any(run_trial), record_diary(opts); end


% ---------------------------------------------------------------------
% 2. load data (only if necessary)
% ---------------------------------------------------------------------
global Dataset
if isfield(Dataset, 'name') && strcmp(Dataset.name, opts.dataset)
    logInfo('Dataset [%s] already loaded', Dataset.name);
elseif any(run_trial)
    datasetFunc  = str2func(['datasets.' opts.dataset]);
    Dataset      = datasetFunc(opts);
    Dataset.name = opts.dataset;
end


% ---------------------------------------------------------------------
% 3. TRAINING: run all _necessary_ trials
% ---------------------------------------------------------------------
logInfo('Training ...');
methodFunc = str2func(['methods.' method]);
methodObj  = methodFunc();
train_online(methodObj, run_trial, opts);
logInfo('%s: Training is done.', opts.identifier);


% ---------------------------------------------------------------------
% 4. TESTING
% ---------------------------------------------------------------------
logInfo('Testing ...');
test_online(res_path, trial_path, run_trial, opts);
logInfo('%s: Testing is done.', opts.identifier);


% ---------------------------------------------------------------------
% 5. Done
% ---------------------------------------------------------------------
logInfo('All done.');
logInfo('Results file: %s', res_path);
logInfo('  Diary file: %s', dia_path);
diary('off');

end
