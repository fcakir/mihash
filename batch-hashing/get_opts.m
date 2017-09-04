function opts = get_opts(opts, dataset, nbits, modelType, varargin)
% Copyright (c) 2017, Fatih Cakir, Kun He, Saral Adel Bargal, Stan Sclaroff 
% All rights reserved.
% 
% If used for academic purposes please cite the below paper:
%
% "MIHash: Online Hashing with Mutual Information", 
% Fatih Cakir*, Kun He*, Sarah Adel Bargal, Stan Sclaroff
% (* equal contribution)
% International Conference on Computer Vision (ICCV) 2017
% 
% Usage of code from authors not listed above might be subject
% to different licensing. Please check with the corresponding authors for
% additioanl information.
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
    logInfo(['creating opts.expDir: ' opts.expDir]);
    mkdir(opts.expDir);
    if ~opts.windows, unix(['chmod g+rw ' opts.expDir]); end
end

end
