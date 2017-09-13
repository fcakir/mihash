function demo_cifar(nbits, modelType, varargin)
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
    
assert(ismember(modelType, {'fc1', 'vggf'}), ...
    'Currently supported model types: {fc1, vggf}');

run ../matconvnet/matlab/vl_setupnn

% init opts
opts = [];
opts.unsupervised = false;
opts.normalize = strcmp(modelType, 'fc1');

opts = opts_batch(opts, 'cifar', nbits, modelType, varargin{:});
logInfo(opts.identifier);

% --------------------------------------------------------------------
%                                               Prepare model and data
% --------------------------------------------------------------------
[net, opts] = get_model(opts);

global imdbType imdb
if ~isempty(imdbType) & strcmp(imdbType, opts.methodID)
    logInfo('IMDB already loaded for %s', imdbType);
else
    imdb = get_imdb(opts);
    imdbType = opts.methodID;
end

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
if strcmp(opts.modelType, 'fc1')
    batchFunc = @batch_fc7;
else
    % imagenet model (VGG)
    imgSize = opts.imageSize;
    meanImage = net.meta.normalization.averageImage;
    if isequal(size(meanImage), [1 1 3])
        meanImage = repmat(meanImage, [imgSize imgSize]);
    else
        assert(isequal(size(meanImage), [imgSize imgSize 3]));
    end
    batchFunc = @(I, B) batch_imagenet(I, B, imgSize, meanImage);
end

% figure out learning rate vector
if opts.lrdecay>0 & opts.lrdecay<1
    % decay by opts.lrdecay every opts.lrstep epochs
    cur_lr = opts.lr;
    lrvec = [];
    while length(lrvec) < opts.epochs
        lrvec = [lrvec, ones(1, opts.lrstep)*cur_lr];
        cur_lr = cur_lr * opts.lrdecay;
    end
else
    % no decay
    lrvec = opts.lr;
end
[net, info] = cnn_train(net, imdb, batchFunc, ...
    'continue'       , opts.continue              , ...
    'debug'          , opts.debug                 , ...
    'plotStatistics' , opts.plot                  , ...
    'expDir'         , opts.expDir                , ...
    'batchSize'      , opts.batchSize             , ...
    'numEpochs'      , opts.epochs                , ...
    'learningRate'   , lrvec                      , ...
    'weightDecay'    , opts.wdecay                , ...
    'val'            , find(imdb.images.set == 3) , ...
    'gpus'           , opts.gpus                  , ...
    'cudnn'          , ~isempty(opts.gpus)        , ...
    'errorFunction'  , 'none') ;

if ~isempty(opts.gpus)
    net = vl_simplenn_move(net, 'gpu'); 
end

% --------------------------------------------------------------------
%                                                                 Test
% --------------------------------------------------------------------
train_id = find(imdb.images.set == 1 | imdb.images.set == 2);
Ytrain   = imdb.images.labels(train_id)';
Htrain   = cnn_encode(net, batchFunc, imdb, train_id, opts)';

test_id  = find(imdb.images.set == 3);
Ytest    = imdb.images.labels(test_id)';
Htest    = cnn_encode(net, batchFunc, imdb, test_id, opts)';

disp('Evaluating...');
opts.metric = 'mAP';
Aff = affinity([], [], Ytrain, Ytest, opts);
evaluate(Htrain, Htest, opts, Aff);

end
