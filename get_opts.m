function opts = get_opts(opts, ftype, dataset, nbits, varargin)
% PARAMS
%  mapping (string) {'smooth', 'bucket', 'coord'}
%  ntrials (int) # of random trials
%  stepsize (float) is step size in SGD
%  SGDBoost (integer) is 0 for OSHEG, 1 for OSH
%  updateInterval (int) update hash table
%  localdir (string) where to save stuff
%  noTrainingPoints (int) # of training points
%  override (int) override previous results {0, 1}
%  tstScenario (int) testing scenario to be used {1 (default -old version),2}

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
ip.addParamValue('val_size', 0, @isscalar);   % set > 0 to d validation
% misc
ip.addParamValue('randseed', 12345, @isscalar);
ip.addParamValue('nworkers', 0, @isscalar);
ip.addParamValue('override', 0, @isscalar);
ip.addParamValue('showplots', 0, @isscalar);
ip.addParamValue('localdir', ...
    '/research/object_detection/cachedir/online-hashing', @isstr);

% Reservoir
ip.addParamValue('reservoirSize', 1000, @isscalar); % reservoir size, set to 0 if reservoir is not used
ip.addParamValue('reg_rs', 0, @isscalar);        % reservoir reg. weight, only for regularization 
ip.addParamValue('reg_smooth', 0, @isscalar);    % smoothness reg. weight
ip.addParamValue('rs_sm_neigh_size',2,@isscalar); % neighbor size for smoothness
ip.addParamValue('sampleResSize',10,@isscalar);   % sample size for reservoir

% hash table update
ip.addParamValue('updateInterval', 100, @isscalar);  % use with baseline
ip.addParamValue('adaptive', 0, @isscalar);        % use with reservoir
ip.addParamValue('trigger', 'mi', @isstr);          % HT update trigger
ip.addParamValue('miThresh', 0, @isscalar);         % for trigger=mi
ip.addParamValue('flipThresh', 0, @isscalar);       % for trigger=bf

% selective bit update
ip.addParamValue('miSelect', 1, @isscalar);  % quality-aware selection strategy {0, 1}
ip.addParamValue('fracHash', 1, @isscalar);  % fraction of hash functions to update (0, 1]
ip.addParamValue('sampleSelectSize',1, @isscalar); % between (0, 1], samples ceil(nbits*sampleSelectSize) 
						 % hash functions to select from during greedy search
ip.addParamValue('miSelectMaxIter', 100, @isscalar); % 
ip.addParamValue('accuHash', 1 ,@isscalar);  % accumulation strategy {0, 1}
ip.addParamValue('randomHash',0, @isscalar); % randomize selected hash functions to be updated {0, 1}
ip.addParamValue('verifyInv',0,@isscalar);   % {0, 1} 

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
if opts.adaptive > 0,
    assert(opts.flipThresh<=0, 'adaptive cannot have flipThresh>0');
end
assert(mod(opts.updateInterval, opts.batchSize) == 0, ...
    sprintf('updateInterval should be a multiple of batchSize(%d)', opts.batchSize));

if ~strcmp(opts.mapping,'smooth')
    opts.updateInterval = opts.noTrainingPoints;
    myLogInfo([opts.mapping ' hashing scheme supports ntests = 2 only' ...
        '\n setting ntests to 2'])
    opts.ntests = 2;
    assert(isempty(opts.methodID)); % OSH only
end

assert(opts.nworkers>=0 && opts.nworkers<=12);
assert(ismember(opts.tstScenario,[1,2]));
assert(opts.fracHash > 0 && opts.fracHash <= 1);
assert(opts.miSelect <= 1);
assert(opts.sampleSelectSize <= 1 && opts.sampleSelectSize > 0);
assert(opts.miSelectMaxIter> 0);

if strcmp(dataset, 'labelme') 
    assert(strcmp(opts.ftype, 'gist'));
    assert(~isempty(opts.methodID)); % if not empty than its not OSH
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

% FC: if mapping is not smooth, set updateInterval to noTrainingPoints
if ~strcmp(opts.mapping, 'smooth') && opts.updateInterval > 0 && ...
        (opts.updateInterval ~= opts.noTrainingPoints)
    myLogInfo('Mapping: %s. Overriding updateInterval=%d to noTrainingPoints=%d', ...
        opts.mapping, opts.updateInterval, opts.noTrainingPoints);
    opts.updateInterval = opts.noTrainingPoints;
end

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
    if opts.reg_rs > 0
        % using reservoir
        idr = sprintf('%sL%g', idr, opts.reg_rs);
    end
    % in this order: U, (F, Ada) or (MI)
    % only possible combinations:  U, U+Ada, F, Ada
    if opts.updateInterval > 0
        idr = sprintf('%sU%d', idr, opts.updateInterval);
    end
    if strcmp(opts.trigger, 'bf')
        if opts.flipThresh > 0
            idr = sprintf('%sF%g', idr, opts.flipThresh);
        end
        if opts.adaptive > 0
            idr = [idr, 'Ada'];
        end
    else
        idr = sprintf('%s-MI%g', idr, opts.miThresh);
    end

    % note: miSelect overrides fracHash
    if opts.miSelect > 0
 	if opts.sampleSelectSize < 1
            idr = sprintf('%s-miSelect%gSample%g-maxIter%g', idr, ...
	    	opts.miSelect, opts.sampleSelectSize, opts.miSelectMaxIter);
        else
	    opts.miSelectMaxIter = opts.nbits;
            idr = sprintf('%s-miSelect%g-maxIter%g', idr, ...
	    	opts.miSelect, opts.miSelectMaxIter);
	end
    else
        if opts.fracHash < 1
            idr = sprintf('%s-frac%g-RND%d-accuHash%d', idr, ...
                opts.fracHash, opts.randomHash, opts.accuHash);
        end
        if opts.verifyInv > 0
            idr = sprintf('%s-inverse', idr);
        end
    end
else
    % no reservoir (baseline): must use updateInterval
    assert(opts.updateInterval > 0);
    idr = sprintf('%s-U%d', idr, opts.updateInterval);
end

% handle smoothness reg
if opts.reg_smooth > 0
    idr = sprintf('%s-SM%gN%dSS%d', idr, opts.reg_smooth, opts.rs_sm_neigh_size, opts.sampleResSize);
end

if 1 %opts.windows
    head = textread('.git/HEAD', '%s');  
    head_ID = textread(['.git/' head{2}], '%s');
    prefix = head_ID{1}(1:7);
    assert(all(isstrprop(prefix, 'xdigit')));
    opts.identifier = [prefix '-' idr];
else
    [~, master_ID] = unix(['git rev-parse --short HEAD']);
    opts.identifier = [master_ID(1:end-1) '-' idr];
end


% -------------------------------------------
% doing validation?
if opts.val_size > 0
    opts.identifier = sprintf('%s-VAL-VS%d', opts.identifier, opts.val_size);
end

% set expdir
expdir_base = sprintf('%s/%s', opts.localdir, opts.identifier);
opts.expdir = sprintf('%s/%gpts_%dtests', expdir_base, opts.noTrainingPoints, opts.ntests);
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
