function demo_cifar(nbits, modelType, varargin)
    
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
