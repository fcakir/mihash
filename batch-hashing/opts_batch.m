function opts = opts_batch(opts, dataset, nbits, modelType, varargin)

ip = inputParser;
ip.addRequired('dataset'    , @isstr);
ip.addRequired('nbits'      , @isscalar);
ip.addRequired('modelType'  , @isstr);

ip.addParameter('split'     , 1);      % train/test split
ip.addParameter('nbins'     , 16);     % # of histogram bins
ip.addParameter('sigscale'  , 40);     % sigmoid scaling factor

ip.addParameter('batchSize' , 100);    % SGD batch size
ip.addParameter('lr'        , 0.1);    % SGD learning rate
ip.addParameter('lrdecay'   , 0.5);    % learning rate decay ratio
ip.addParameter('lrstep'    , 10);     % learning rate decay step
ip.addParameter('wdecay'    , 0.0005); % weight decay
ip.addParameter('epochs'    , 100);    % # of epochs

ip.addParameter('gpus'      , []);
ip.addParameter('continue'  , true);   % continue from saved model
ip.addParameter('debug'     , false);
ip.addParameter('plot'      , false);

ip.KeepUnmatched = true;
ip.parse(dataset, nbits, modelType, varargin{:});
opts = catstruct(ip.Results, opts);


% post processing

opts.methodID = sprintf('deepMI-%s-sp%d', opts.dataset, opts.split);

prefix = sprintf('%s',datetime('today','Format','yyyyMMdd')); 
opts.identifier = sprintf('%s-%snorm%d-%dbit-Bins%dSig%g-Batch%d-LR%gD%gS%d', ...
    prefix, opts.modelType, opts.normalize, opts.nbits, opts.nbins, opts.sigscale, ...
    opts.batchSize, opts.lr, opts.lrdecay, opts.lrstep);

opts.dataDir  = '../data';
opts.localDir = '../cachedir';
if ~exist(opts.localDir)
    error('../cachedir does not exist!');
end
exp_base = fullfile(opts.localDir, opts.methodID);
opts.expDir = fullfile(exp_base, opts.identifier);
if ~exist(exp_base, 'dir'), mkdir(exp_base); end
if ~exist(opts.expDir, 'dir'),
    logInfo(['creating expDir: ' opts.expDir]);
    mkdir(opts.expDir);
end

disp(opts);

end
