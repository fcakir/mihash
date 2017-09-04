function opts = get_opts(opts, dataset, nbits, varargin)
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
% to different licensing. Please check with the corresponding authors for
% additional information.
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
% Sets up data stream settings.
% The data stream constitutes "trainingPoints" x "epochs" training instances, i.e., the
% online learning is continued until "trainingPoints" x "epochs" examples are processed
% where "trainingPoints" should be smaller than the available training data.  
% In this data stream, "ntests" checkpoints are randomly placed in which the 
% performance, as measured by the "metric" value, is evaluated. This can be used
% to plot performance vs. training instances curves. See illustration below.
%
% |----o-------o--------o----o-------o--------| ← Data stream 
% ↑    ↑                ↑                     ↑
% 1 "checkpoint #2"  "checkpoint #4"        "trainingPoints" x "epochs"
%
% (checkpoint #1 is on the 1st iteration)
% 
% REFERENCES
% [1] Fatih Cakir*, Kun He*, Sarah Adel Bargal, Stan Sclaroff (*equal contribution)
%     "MIHash: Online Hashing with Mutual Information", 
%     International Conference on Computer Vision (ICCV) 2017.
%
% INPUTS
% 	dataset - (string) A string denoting the dataset to be used. 
%			   Please add/edit  load_gist.m and load_cnn.m for available
%			   datasets.
% 	nbits   - (int)    Hash code length
% 	
% noTrainingPoints - (int) The number of training instances to be process at each
% 			   epoch. Must be smaller than the available training data.
% 	ntrials - (int)	   The number of trials. A trial corresponds to a separate
% 			   experimental run. The performance results then are average
% 			   across different trials. See test.m .
% 	ntests  - (int)    This parameter corresponds to the number of checkpoints
% 			   as illustrated above. This amount of checkpoints is placed
% 			   at random locations in the data stream to evaluate the 
% 			   hash methods performance. Must be [2, "trainingPoints"x"epochs"].
% 			   The performance is evaluated on the first and last 
%			   iteration, at the least.
% 	metric  - (string) Choices are 'prec_kX', 'prec_nX', 'mAP_X' and 'mAP' where
%			   X is an integer. 'prec_k100' evaluates the  precision 
% 			   value of the 100 nearest neighbors (average over the 
% 			   query set). 'prec_n3' evaluates the precision value 
% 			   of neighbors returned within a Hamming radius 3 
% 			   (averaged over the query set). 'mAP_100' evaluates the
% 			   average precision of the 100 nearest neighbors (average
%			   over the query set). 'mAP' evaluates the mean average 
% 			   precision.  
%	epoch 	- (int)    Number of epochs, [1, inf) 			    
% 	prefix 	- (string) Prefix for the results folder title, if empty, the 
% 			   results folder will be prefixes with todays date.
%  randseed - (int)   Random seed for reproducility. 			   
%  nworkers - (int)   Number of parallel workers. If ntrials > 1, each trial
% 			   is run on a different worker. Testing is done in a 
% 			   parallel manner as well. 
% 	override - (int)   {0, 1}. If override=0, then training and/or testing
% 			   that correspond to the same experiment, is avoided when
% 			   possible. if override=1, then (re-)runs the experiment
% 			   no matter what.
%      showplots - (int)   {0, 1}. If showplots=1, plots the performance curve wrt
% 			   training instances, CPU time and Bit Recomputations. 
% 			   See test.m .
%      localdir - (string) Directory path where the results folder will be created.
% reservoirSize - (int)    The size of the set to be sampled via reservoir sampling 
% 			   from the data stream. The reservoir can be used to 
% 			   compute statistical properties of the stream. For future 
% 			   release purposes.
% updateInterval - (int)   The hash table is updated after each "updateInterval" 
%			   number of training instances is processed.
%	trigger	- (string) Choices are 'bf', 'mi' or 'fix' (default) only. 
% 			   The type of trigger used to determine if a hash table update is 
% 			   needed. 'bf' means bit flips, 'mi' means mutual information (see 
% 			   reference [1] above), and 'fix' means no trigger. If 'fix' is selected, then
% 			   the hash table is updated at every 'updateInterval'.
%     flipThresh - (int)   If the amount of bit flips in the reservoir hash table 
% 			   exceeds "flipThresh", a hash table update is performed. 
% 			   Evaluated only after each "updateInterval". If flipThresh=-1
% 			   the hash table is always updated at each "updateInterval".
% 			   Hard-coded to 0. For future release.
% 
% OUTPUTS
%	opts	- (struct) struct containing name and values of the inputs, e.g.,
%			   opts.dataset, opts.nbits, ... .

ip = inputParser;

% train/test
ip.addRequired('dataset', @isstr);
ip.addRequired('nbits', @isscalar);
ip.addParamValue('noTrainingPoints', 20e3, @isscalar);
ip.addParamValue('ntrials', 3, @isscalar);
ip.addParamValue('ntests', 50, @isscalar);
ip.addParamValue('metric', 'mAP', @isstr);    % evaluation metric
ip.addParamValue('epoch', 1, @isscalar)
% misc
ip.addParamValue('prefix','', @isstr);
ip.addParamValue('randseed', 12345, @isscalar);
ip.addParamValue('nworkers', 0, @isscalar);
ip.addParamValue('override', 0, @isscalar);
ip.addParamValue('showplots', 0, @isscalar);
ip.addParamValue('localdir', './cachedir', @isstr);

% Reservoir
ip.addParamValue('reservoirSize', 0, @isscalar); % reservoir size, set to 0 if reservoir is not used

% hash table update
ip.addParamValue('updateInterval', 100, @isscalar);  % use with baseline
ip.addParamValue('trigger', 'mi', @isstr);          % HT update trigger
ip.addParamValue('miThresh', 0, @isscalar);       % for trigger=mi
ip.addParamValue('flipThresh', -1, @isscalar);       % for trigger=bf

% parse input
ip.KeepUnmatched = true;
ip.parse(lower(dataset), nbits, varargin{:});
opts = catstruct(ip.Results, opts);  % combine w/ existing opts


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ASSERTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
assert(opts.ntests >= 2, 'ntests should be at least 2 (first & last iter)');
assert(mod(opts.updateInterval, opts.batchSize) == 0, ...
    sprintf('updateInterval should be a multiple of batchSize(%d)', opts.batchSize));

assert(opts.nworkers>=0 && opts.nworkers<=12);

if strcmp(opts.dataset, 'labelme') 
    assert(~strcmpi(opts.methodID,'osh')); % OSH is inapplicable on LabelMe 
    opts.unsupervised = 1;
    opts.ftype = 'gist';
else
    opts.unsupervised = 0;
    opts.ftype = 'cnn';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CONFIGS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% localdir
if isfield(opts, 'methodID') && ~isempty(opts.methodID)
    opts.localdir = fullfile(opts.localdir, opts.methodID);
end
if exist(opts.localdir, 'dir') == 0, 
    mkdir(opts.localdir);
end

% matlabpool handling
if isempty(gcp('nocreate')) && opts.nworkers > 0
    logInfo('Opening parpool, nworkers = %d', opts.nworkers);
    delete(gcp('nocreate'))  % clear up zombies
    p = parpool(opts.nworkers);
end

% set randseed -- don't change the randseed if don't have to!
rng(opts.randseed, 'twister');

% decipher evaluation metric
if ~isempty(strfind(opts.metric, 'prec_k'))
    % eg. prec_k3 is precision at k=3
    opts.prec_k = sscanf(opts.metric(7:end), '%d');
elseif ~isempty(strfind(opts.metric, 'prec_n'))
    % eg. prec_n3 is precision at n=3
    opts.prec_n = sscanf(opts.metric(7:end), '%d');
elseif ~isempty(strfind(opts.metric, 'mAP_'))
    % eg. mAP_1000 is mAP @ top 1000 retrievals
    opts.mAP = sscanf(opts.metric(5:end), '%d');
else 
    % default: mAP
    assert(strcmp(opts.metric, 'mAP'), ['unknown opts.metric: ' opts.metric]);
end


% --------------------------------------------
% identifier string for the current experiment
% NOTE: opts.identifier is already initialized with method-specific params
opts.identifier = sprintf('%s-%s-%d-%s', opts.dataset, opts.ftype, ...
    opts.nbits, opts.identifier);
idr = opts.identifier;

% handle reservoir
if opts.reservoirSize > 0
    idr = sprintf('%s-RS%d', idr, opts.reservoirSize);
    % in this order: U, (F, Ada) or (MI)
    % only possible combinations:  U, U+Ada, F, Ada
    if opts.updateInterval > 0
        idr = sprintf('%sU%d', idr, opts.updateInterval);
    end
    if strcmp(opts.trigger, 'bf')
        if opts.flipThresh > 0
            idr = sprintf('%sF%g', idr, opts.flipThresh);
        end
    elseif strcmp(opts.trigger, 'mi')
		idr = sprintf('%s-MI%g', idr, opts.miThresh);
    else
        idr = sprintf('%s-FIX_Update', idr);
    end
else
    % no reservoir (baseline): must use updateInterval
    assert(opts.updateInterval > 0);
    idr = sprintf('%s-U%d', idr, opts.updateInterval);
end
if isempty(opts.prefix), prefix = sprintf('%s',datetime('today','InputFormat','yyyy-MM-dd')); end;
opts.identifier = [prefix '-' idr];

% set expdir
expdir_base = sprintf('%s/%s', opts.localdir, opts.identifier);
opts.expdir = sprintf('%s/%gpts_%gepochs_%dtests', expdir_base, opts.noTrainingPoints*opts.epoch, opts.epoch, opts.ntests);
if ~exist(expdir_base, 'dir'), mkdir(expdir_base); end
if ~exist(opts.expdir, 'dir'),
    logInfo(['creating opts.expdir: ' opts.expdir]);
    mkdir(opts.expdir);
end

% FINISHED
logInfo('identifier: %s', opts.identifier);
disp(opts);
end
