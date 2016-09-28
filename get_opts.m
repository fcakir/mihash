function opts = get_opts(ftype, dataset, nbits, varargin)
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
% default values
ip.addParamValue('ftype', ftype, @isstr);
ip.addParamValue('dataset', dataset, @isstr);
ip.addParamValue('nbits', nbits, @isscalar);

ip.addParamValue('mapping', 'smooth', @isstr);
ip.addParamValue('stepsize', 0.1, @isscalar);
ip.addParamValue('SGDBoost', 0, @isscalar);
ip.addParamValue('randseed', 12345, @isscalar);
ip.addParamValue('localdir', ...
    '/research/object_detection/cachedir/online-hashing', @isstr);
ip.addParamValue('noTrainingPoints', 20000, @isscalar);
ip.addParamValue('override', 0, @isscalar);
ip.addParamValue('showplots', 1, @isscalar);

ip.addParamValue('nworkers', 6, @isscalar);
ip.addParamValue('ntrials', 5, @isscalar);
ip.addParamValue('ntests', 10, @isscalar);
ip.addParamValue('testFrac', 1, @isscalar);  % <1 for faster testing
ip.addParamValue('metric', 'mAP', @isstr);    % evaluation metric

% controling when to update hash table
% default: save every opts.updateInterval iterations
% IF use reservoir AND opts.flipThresh > 0, THEN use opts.flipThresh
ip.addParamValue('updateInterval', -1, @isscalar);  % use with baseline
ip.addParamValue('flipThresh', -1, @isscalar);      % use with reservoir
ip.addParamValue('adaptive', -1, @isscalar);        % use with reservoir
ip.addParamValue('trigger', 'bf', @isstr);          % HT update trigger

ip.addParamValue('reservoirSize', 50, @isscalar); % reservoir size, set to 0 if reservoir is not used
ip.addParamValue('reg_rs', -1, @isscalar);        % reservoir reg. weight, only for regularization 
ip.addParamValue('reg_smooth', -1, @isscalar);    % smoothness reg. weight
ip.addParamValue('rs_sm_neigh_size',2,@isscalar); % neighbor size for smoothness
ip.addParamValue('sampleResSize',10,@isscalar);   % sample size for reservoir
ip.addParamValue('miThresh', 0, @isscalar);       % threshold for the mutual info
ip.addParamValue('fracHash', 1, @isscalar);       % fraction of hash functions to update (0, 1]
ip.addParamValue('accuHash', 1 ,@isscalar);       % accumulation strategy {0, 1}
ip.addParamValue('randomHash',0, @isscalar);      % randomize selected hash functions to be updated {0, 1}
ip.addParamValue('verifyInv',0,@isscalar);        % {0, 1} 
% Hack for Places
ip.addParamValue('labelspercls', 0, @isscalar);

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
ip.parse(varargin{:});
opts = ip.Results;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% assertions
assert(ismember(opts.ftype, {'gist', 'cnn'}));
assert(opts.testFrac > 0);
assert(opts.ntests >= 2, 'ntests should be at least 2 (first & last iter)');
assert(~(opts.updateInterval>0 && opts.flipThresh>0), ...
    'updateInterval cannot be used with flipThresh');
if opts.adaptive > 0,
    assert(opts.flipThresh<=0, 'adaptive cannot have flipThresh>0');
end

if ~strcmp(opts.mapping,'smooth')
    opts.updateInterval = opts.noTrainingPoints;
    myLogInfo([opts.mapping ' hashing scheme supports ntests = 2 only' ...
        '\n setting ntests to 2'])
    opts.ntests = 2;
end

assert(opts.nworkers>=0 && opts.nworkers<=12);
assert(ismember(opts.tstScenario,[1,2]));
assert(opts.fracHash > 0 && opts.fracHash <= 1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% are we on window$?
opts.windows = ~isempty(strfind(computer, 'WIN'));
if opts.windows
    % reset localdir
    opts.localdir = '\\kraken\object_detection\cachedir\online-hashing';
    myLogInfo('We are on Window$. localdir set to %s', opts.localdir);
end

% matlabpool handling
if isempty(gcp('nocreate')) && opts.nworkers > 0
    myLogInfo('Opening matlabpool, nworkers = %d', opts.nworkers);
    delete(gcp('nocreate'))  % clear up zombies
    p = parpool(opts.nworkers);
end

% make localdir
if ~exist(opts.localdir, 'dir'),
    mkdir(opts.localdir);
    if ~opts.windows, unix(['chmod g+rw ' opts.localdir]); end
end

% set randseed -- don't change the randseed if don't have to!
rng(opts.randseed);

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
    if opts.labelspercls > 0
        assert(opts.labelspercls >= 500 && opts.labelspercls <= 5000, ...
            'please give a reasonable labelspercls in [500, 5000]');
        myLogInfo('Places will use %d labeled examples per class', opts.labelspercls);
        opts.dataset = [opts.dataset, 'L', num2str(opts.labelspercls)];
    else
        myLogInfo('Places: fully supervised experiment');
    end
end

% identifier string for the current experiment
opts.identifier = sprintf('%s-%s-%d%s-B%dS%g', opts.dataset, opts.ftype, ...
    opts.nbits, opts.mapping, opts.SGDBoost, opts.stepsize);
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
    if opts.fracHash < 1
	idr = sprintf('%s-frac%g-RND%d-accuHash%d', idr, opts.fracHash, opts.randomHash, opts.accuHash);
    end
    if opts.verifyInv > 0
    	idr = sprintf('%s-inverse', idr);
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

opts.identifier = idr;
myLogInfo('identifier: %s', opts.identifier);

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

% decipher evaluation metric
if ~isempty(strfind(opts.metric, 'prec_k'))
    % eg. prec_k3 is precision at k=3
    opts.prec_k = sscanf(opts.metric(7:end), '%d');
elseif ~isempty(strfind(opts.metric, 'prec_n'))
    % eg. prec_n3 is precision at n=3
    opts.prec_n = sscanf(opts.metric(7:end), '%d');
elseif ~isempty(strfind(opts.metric, 'mAP_'))
    % eg. mAP_1000 is mAP @ top 1000 retrievals
    opts.mAP = sscanf(opts.metric(5:end), '%d')
else 
    % default: mAP
    assert(strcmp(opts.metric, 'mAP'), ['unknown opts.metric: ' opts.metric]);
end

% hold a diary -save it to opts.expdir
if opts.override
    unix(['rm -f ' opts.expdir '/diary*']);
end
diary_index = 1;
opts.diary_name = sprintf('%s/diary_%03d.txt', opts.expdir, diary_index);
while exist(opts.diary_name,'file') && ~opts.override
    diary_index = diary_index + 1;
    opts.diary_name = sprintf('%s/diary_%03d.txt', opts.expdir, diary_index);
end

diary(opts.diary_name);
diary('on');
% FINISHED
disp(opts);
end
