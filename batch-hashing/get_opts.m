function opts = get_opts(opts, dataset, nbits, modelType, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ip = inputParser;
ip.addRequired('dataset', @isstr);
ip.addRequired('nbits', @isscalar);
ip.addRequired('modelType', @isstr);

ip.addParameter('metric', 'mAP');
ip.addParameter('randseed', 12345);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MatConvNet specific
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.networkType = 'simplenn' ;
opts.train = struct() ;

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

opts.localDir = '../cachedir';  % symlink
if ~exist(opts.localDir, 'file')
    error('Please make a symlink for cachedir!');
end
opts.dataDir = fullfile(opts.localDir, 'data');
opts.imdbPath = fullfile(opts.dataDir, [opts.dataset '_imdb']);

% expDir: format like [localDir]/deepMI-cifar32-fc
opts.expDir = fullfile(opts.localDir, opts.methodID);
if exist(opts.expDir, 'dir') == 0, 
    mkdir(opts.expDir);
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
    logInfo(['creating opts.expDir: ' opts.expDir]);
    mkdir(opts.expDir);
end

end
