function opts = get_opts(opts, ftype, dataset, nbits, varargin)
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
% INPUTS
% 	ftype	- (string) Choices are 'gist' and 'cnn'. load_gist.m and load_cnn.m
%			   function are called. 'gist' and 'cnn' correspond to
% 			   GIST and CNN descriptors, respectively.% 			   
% 			   Please inspect/edit load_gist.m and load_cnn.m for
% 			   further information. 
% 	dataset - (string) A string denoting the dataset to be used. 
%			   Please add/edit  load_gist.m and load_cnn.m for available
%			   datasets.
% 	nbits   - (int)    Hash code length
% 	mapping - (string) Choices are 'smooth' and 'bucket'. 'Smooth' populates
% 			   the hash table with the hash mapping output. 'bucket'
% 			   is only applicate for the "Online Supervised Hashing (osh)"
% 			   method in which the hash table can be populated with 
% 			   Error Correcting Output Codes of the data items (if their
%			   label information exists). 'smooth' is the traditional
% 			   approach in hashing methods. 
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
%      testFrac - (float)  A value between (0, 1]. testFrac = 0.5 results in only
%			   testing with a random half of the test/query set. For
% 			   speed purposes. 
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
% 	no_blocks - (int)  Hard-coded to 1. For future release.
%	randseed - (int)   Random seed for reproducility. 			   
%	nworkers - (int)   Number of parallel workers. If ntrials > 1, each trial
% 			   is run on a different worker. Testing is done in a 
% 			   parallel manner as well. 
% 	override - (int)   {0, 1}. If override=0, then training and/or testing
% 			   that correspond to the same experiment, is avoided when
% 			   possible. if override=1, then (re-)runs the experiment
% 			   no matter what.
%	val_size - (int)   {0, 1}. Should be kept to 0, for future release purposes.
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
%	trigger	- (string) Choices are 'bf' only. The type of trigger used to determine 
% 			   if a hash table update is needed. 'bf' means bit flips.
% 			   For future release purposes.
%     flipThresh - (int)   If the amount of bit flips in the reservoir hash table 
% 			   exceeds "flipThresh", a hash table update is performed. 
% 			   Evaluated only after each "updateInterval". If flipThresh=-1
% 			   the hash table is always updated at each "updateInterval".
% 			   Hard-coded to 0. For future release.
%   labelsPerCls - (int)   Hard-coded to 0. For future release.
%   tstScenario  - (int)   Hard-coded to 1. Corresponds to populating the hash table 
% 			   with all the data, excluding the test set. Other
% 			   alternative might be to populate only with the processed/observed
% 			   training instances. 
%       pObserve - (float) For multiclass datasets. To generate different data streams
% 			   in which a new class appears with pObserve probability. 
%			   pObserve=0 corresponds to uniform probability, i.e., 
% 			   see get_ordering.m .
% 
% OUTPUTS
%	opts	- (struct) struct containing name and values of the inputs, e.g.,
%			   opts.ftype, opts.dataset, opts.nbits, ... .

ip = inputParser;

% train/test
ip.addRequired('ftype', @isstr);
ip.addRequired('dataset', @isstr);
ip.addRequired('nbits', @isscalar);
ip.addParamValue('mapping', 'smooth', @isstr);
ip.addParamValue('noTrainingPoints', 20e3, @isscalar);
ip.addParamValue('ntrials', 3, @isscalar);
ip.addParamValue('ntests', 50, @isscalar);
ip.addParamValue('testFrac', 1, @isscalar);  % <1 for faster testing
ip.addParamValue('metric', 'mAP', @isstr);    % evaluation metric
ip.addParamValue('epoch', 1, @isscalar)
% misc
ip.addParamValue('prefix','', @isstr);
ip.addParamValue('no_blocks', 1, @isscalar);
ip.addParamValue('randseed', 12345, @isscalar);
ip.addParamValue('nworkers', 0, @isscalar);
ip.addParamValue('override', 0, @isscalar);
ip.addParamValue('val_size', 0, @isscalar);
ip.addParamValue('showplots', 0, @isscalar);
ip.addParamValue('localdir', ...
    '/research/object_detection/cachedir/online-hashing', @isstr);

% Reservoir
ip.addParamValue('reservoirSize', 0, @isscalar); % reservoir size, set to 0 if reservoir is not used

% hash table update
ip.addParamValue('updateInterval', 100, @isscalar);  % use with baseline
ip.addParamValue('trigger', 'bf', @isstr);          % HT update trigger
ip.addParamValue('flipThresh', -1, @isscalar);       % for trigger=bf

% Hack for Places
ip.addParamValue('labelsPerCls', 0, @isscalar);

% Testing scenario
% TODO explain
ip.addParamValue('tstScenario',1,@isscalar);

% for label arrival strategy: prob. of observing a new label
% NOTE: if pObserve is too small then it may exhaust examples in some class
%       before getting a new label
% - For CIFAR  0.002 seems good (observe all labels @~5k)
% - For PLACES 0.025 (observe all labels @~9k)
%              0.05  (observe all labels @~5k)
ip.addParamValue('pObserve', 0, @isscalar);

% parse input
ip.KeepUnmatched = true;
ip.parse(ftype, dataset, nbits, varargin{:});
opts = catstruct(ip.Results, opts);  % combine w/ existing opts


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ASSERTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
assert(ismember(opts.ftype, {'gist', 'cnn'}));
assert(opts.testFrac > 0);
assert(opts.ntests >= 2, 'ntests should be at least 2 (first & last iter)');
%assert(~(opts.updateInterval>0 && opts.flipThresh>0), ...
    %'updateInterval cannot be used with flipThresh');
assert(mod(opts.updateInterval, opts.batchSize) == 0, ...
    sprintf('updateInterval should be a multiple of batchSize(%d)', opts.batchSize));

if ~strcmp(opts.mapping,'smooth')
    opts.updateInterval = opts.noTrainingPoints;
    myLogInfo([opts.mapping ' hashing scheme supports ntests = 2 only' ...
        '\n setting ntests to 2'])
    opts.ntests = 2;
    assert(strcmpi(opts.methodID,'osh')); % OSH only
end

assert(opts.nworkers>=0 && opts.nworkers<=12);
assert(ismember(opts.tstScenario,[1,2]));

if strcmp(dataset, 'labelme') 
    assert(strcmp(opts.ftype, 'gist'));
    assert(~strcmpi(opts.methodID,'osh')); % OSH is inapplicable on LabelMe 
    opts.unsupervised = 1;
else
    opts.unsupervised = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CONFIGS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% are we on window$?
opts.windows = ~isempty(strfind(computer, 'WIN'));
if opts.windows
    % reset localdir
    opts.localdir = '\\kraken\object_detection\cachedir\online-hashing';
    myLogInfo('We are on Window$. localdir set to %s', opts.localdir);
end
% localdir
if isfield(opts, 'methodID') && ~isempty(opts.methodID)
    opts.localdir = fullfile(opts.localdir, opts.methodID);
end
if exist(opts.localdir, 'dir') == 0, 
    mkdir(opts.localdir);
    if ~opts.windows, unix(['chmod g+rw ' opts.localdir]); end
end

% matlabpool handling
if isempty(gcp('nocreate')) && opts.nworkers > 0
    myLogInfo('Opening parpool, nworkers = %d', opts.nworkers);
    delete(gcp('nocreate'))  % clear up zombies
    p = parpool(opts.nworkers);
end

% set randseed -- don't change the randseed if don't have to!
rng(opts.randseed, 'twister');

% if smoothness not applied set sample reservoir size to the entire reservoir
% [hack] for places
if strcmp(opts.dataset, 'places')
    if opts.labelsPerCls > 0
        assert(opts.labelsPerCls >= 500 && opts.labelsPerCls <= 5000, ...
            'please give a reasonable labelsPerCls in [500, 5000]');
        myLogInfo('Places will use %d labeled examples per class', opts.labelsPerCls);
        opts.dataset = [opts.dataset, 'L', num2str(opts.labelsPerCls)];
    else
        myLogInfo('Places: fully supervised experiment');
    end
end

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
opts.identifier = sprintf('%s-%s-%d%s-%s', opts.dataset, opts.ftype, ...
    opts.nbits, opts.mapping, opts.identifier);
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
if opts.pObserve > 0
    assert(opts.pObserve < 1);
    opts.expdir = sprintf('%s_arr%g', opts.expdir, opts.pObserve);
end
if opts.tstScenario == 2
    opts.expdir = sprintf('%s_scenario%d', opts.expdir, opts.tstScenario);
end
if ~exist(expdir_base, 'dir'),
    mkdir(expdir_base);
    if ~opts.windows, unix(['chmod g+rw ' expdir_base]); end
end
if ~exist(opts.expdir, 'dir'),
    myLogInfo(['creating opts.expdir: ' opts.expdir]);
    mkdir(opts.expdir);
    if ~opts.windows, unix(['chmod g+rw ' opts.expdir]); end
end

% hold a diary -save it to opts.expdir
if opts.override
    try
        unix(['rm -f ' opts.expdir '/diary*']);
    end
end
diary_index = 1;
opts.diary_name = sprintf('%s/diary_%03d.txt', opts.expdir, diary_index);
while exist(opts.diary_name,'file') % && ~opts.override
    diary_index = diary_index + 1;
    opts.diary_name = sprintf('%s/diary_%03d.txt', opts.expdir, diary_index);
end
diary(opts.diary_name);
diary('on');

% FINISHED
myLogInfo('identifier: %s', opts.identifier);
disp(opts);
end
