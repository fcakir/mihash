function demo_cifar(nbits, modelType, varargin)
    
addpath(fullfile(pwd, '..'));
addpath(fullfile(pwd, '..', 'util'));
%run ./vlfeat/toolbox/vl_setup
run ./matconvnet_gpu/matlab/vl_setupnn

% init opts
ip = inputParser;
ip.addParameter('split', 1);
ip.addParameter('nbins', nbits);

ip.addParameter('obj', 'mi');  % mi or fastap

ip.addParameter('batchSize', 100);
ip.addParameter('solver', 'sgd');
ip.addParameter('lr', 0.1);
ip.addParameter('lrdecay', 0);
ip.addParameter('lrstep', 10);
ip.addParameter('wdecay', 0.0005);
ip.addParameter('bpdepth', Inf);  % default: update all layers
ip.addParameter('sigmf_p', 40);

ip.addParameter('gpus', []);
ip.addParameter('normalize', true);
ip.addParameter('continue', false);
ip.addParameter('debug', false);
ip.addParameter('plot', false);
ip.KeepUnmatched = true;
ip.parse(varargin{:});
opts = ip.Results;
opts.methodID = sprintf('%s-cifar%d-sp%d-%s', upper(opts.obj), nbits, ...
    opts.split, modelType);
opts.identifier = sprintf('%dbins-batch%d-%sLR%gD%gE%d-Sig%d', ...
    opts.nbins, opts.batchSize, opts.solver, opts.lr, opts.lrdecay, opts.lrstep, ...
    opts.sigmf_p);
assert(ismember(modelType, {'fc1', 'vggf'}), 'Supported model types: fc1, vggf');
opts.normalize = strcmp(modelType, 'fc1');
if ~opts.normalize
    opts.identifier = [opts.identifier, '-nonorm']; 
end

opts = get_opts(opts, 'cifar', nbits, modelType, varargin{:})
opts.unsupervised = false;
disp(opts.identifier);

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
    while length(lrvec) < opts.epoch
        lrvec = [lrvec, ones(1, opts.lrstep)*cur_lr];
        cur_lr = cur_lr * opts.lrdecay;
    end
else
    % no decay
    lrvec = opts.lr;
end
[net, info] = cnn_train(net, imdb, batchFunc, ...
    'continue', opts.continue, ...
    'debug', opts.debug, ...
    'plotStatistics', opts.plot, ...
    'expDir', opts.expDir, ...
    'batchSize', opts.batchSize, ...
    'numEpochs', opts.epoch, ...
    'learningRate', lrvec, ...
    'weightDecay', opts.wdecay, ...
    'backPropDepth', opts.bpdepth, ...
    'val', find(imdb.images.set == 3), ...
    'gpus', opts.gpus, ...
    'errorFunction', 'none') ;

if ~isempty(opts.gpus)
    net = vl_simplenn_move(net, 'gpu'); 
end

% --------------------------------------------------------------------
%                                                                 Test
% --------------------------------------------------------------------
train_id = find(imdb.images.set == 1 | imdb.images.set == 2);
Ytrain   = imdb.images.labels(train_id)';
Htrain   = cnn_encode(net, batchFunc, imdb, train_id, opts);

test_id = find(imdb.images.set == 3);
Ytest   = imdb.images.labels(test_id)';
Htest   = cnn_encode(net, batchFunc, imdb, test_id, opts);

disp('Evaluating...');
opts.metric = 'mAP';
opts.unsupervised = false;
evaluate(Htrain, Htest, Ytrain, Ytest, opts);
end
