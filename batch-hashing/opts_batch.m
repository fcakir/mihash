function opts = opts_batch(opts, dataset, nbits, modelType, varargin)
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
