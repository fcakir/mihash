function opts = opts_matconvnet(opts, dataset, nbits, modelType, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ip = inputParser;
ip.addRequired('dataset', @isstr);
ip.addRequired('nbits', @isscalar);
ip.addRequired('modelType', @isstr);

ip.addParameter('metric', 'mAP');
ip.addParameter('epoch', 1);
ip.addParameter('randseed', 12345);

% misc
ip.addParameter('mapping', 'smooth');
ip.addParameter('ntests', 50);
ip.addParameter('override', 0);
ip.addParameter('showplots', 0);
ip.addParameter('val_size', 0);
ip.addParameter('updateInterval', 100);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MatConvNet specific
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.networkType = 'simplenn' ;
opts.train = struct() ;
%opts = vl_argparse(opts, varargin) ;

if ~isfield(opts.train, 'gpus'), opts.train.gpus = opts.gpus; end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% parse input
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ip.KeepUnmatched = true;
ip.parse(dataset, nbits, modelType, varargin{:});
opts = catstruct(ip.Results, opts);  % combine w/ existing opts


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% post-parse processing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opts.windows = ~isempty(strfind(computer, 'WIN'));
if opts.windows
    opts.localDir = '\\kraken\object_detection\cachedir\deep-hashing';
    myLogInfo('We are on Window$. localDir set to %s', opts.localDir);
else
    %opts.localDir = '/research/object_detection/cachedir/deep-hashing';
    opts.localDir = './cachedir';  % use symlink on linux
    if ~exist(opts.localDir, 'file')
        error('Please make a symlink for cachedir!');
    end
end
opts.dataDir = fullfile(opts.localDir, 'data');
opts.imdbPath = fullfile(opts.dataDir, [opts.dataset '_imdb']);

% expDir: format like .../deep-hashing/deepMI-cifar32-fc
opts.expDir = fullfile(opts.localDir, opts.methodID);
if exist(opts.expDir, 'dir') == 0, 
    mkdir(opts.expDir);
    if ~opts.windows, unix(['chmod g+rw ' opts.localDir]); end
end

% evaluation metric
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
idr = opts.identifier;

head = textread('../.git/HEAD', '%s');  
head_ID = textread(['../.git/' head{2}], '%s');
prefix = head_ID{1}(1:7);
assert(all(isstrprop(prefix, 'xdigit')));
opts.identifier = [prefix '-' idr];
% --------------------------------------------

% expand expDir
% expDir (orig): .../deep-hashing/deepMI-cifar32-fc
% identifier: abcdef-maxdif0.1-......
opts.expDir = fullfile(opts.expDir, opts.identifier);
if ~exist(opts.expDir, 'dir'),
    myLogInfo(['creating opts.expDir: ' opts.expDir]);
    mkdir(opts.expDir);
    if ~opts.windows, unix(['chmod g+rw ' opts.expDir]); end
end

end
