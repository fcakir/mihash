function demo_cifar(nbits, modelType, varargin)
    
addpath(fullfile(pwd, '..'));
addpath(fullfile(pwd, '..', 'util'));
run ./vlfeat/toolbox/vl_setup
run ./matconvnet_gpu/matlab/vl_setupnn

% init opts
ip = inputParser;
ip.addParameter('split', 1);
ip.addParameter('nbins', nbits);

ip.addParameter('obj', 'mi');  % mi or fastap
ip.addParameter('maxdif', 0);
ip.addParameter('minplus', 0);
ip.addParameter('quant', 0);

ip.addParameter('batchSize', 100);
ip.addParameter('solver', 'sgd');
ip.addParameter('lr', 0.1);
ip.addParameter('lrdecay', 0);
ip.addParameter('lrepoch', 10);
ip.addParameter('wdecay', 0);
ip.addParameter('bpdepth', 8);  % up to conv5 for VGG
%ip.addParameter('momentum', 0.9);
ip.addParameter('sigmf_p', [40 0]);

ip.addParameter('gpus', []);
ip.addParameter('normalize', true);
ip.addParameter('continue', false);
ip.addParameter('debug', false);
ip.addParameter('plot', false);
ip.KeepUnmatched = true;
ip.parse(varargin{:});
opts = ip.Results;
opts.methodID = sprintf('deep%s-cifar%d-sp%d-%s', upper(opts.obj), nbits, ...
    opts.split, modelType);
opts.identifier = sprintf('%dbins-maxdif%g-batch%d-%sLR%gD%gE%d-Sig%d_%d', ...
    opts.nbins, opts.maxdif, ...
    opts.batchSize, ...
    opts.solver, opts.lr, opts.lrdecay, opts.lrepoch, ...
    opts.sigmf_p(1), opts.sigmf_p(2));
if opts.minplus > 0
    opts.identifier = sprintf('%s-minplus%g', opts.identifier, opts.minplus);
end
if ismember(modelType, {'alexnet', 'vgg16', 'vggf'})
    opts.normalize = false;
end
if ~opts.normalize
    opts.identifier = [opts.identifier, '-nonorm']; 
end

opts = opts_deepMI(opts, 'cifar', nbits, modelType, varargin{:})
opts.unsupervised = false;
disp(opts.identifier);

% --------------------------------------------------------------------
%                                               Prepare model and data
% --------------------------------------------------------------------
[net, opts] = get_model(opts);

global imdbType imdb
if ~isempty(imdbType) & strcmp(imdbType, opts.methodID)
    myLogInfo('IMDB already loaded for %s', imdbType);
else
    imdb = get_imdb(opts);
    imdbType = opts.methodID;
end

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
if ismember(opts.modelType, {'fc', 'fc1'})
    batchFunc = @batch_fc7;
elseif strcmp(opts.modelType, 'nin')
    % NIN on orig images
    batchFunc = @batch_simplenn;
else
    % imagenet model
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
    cur_lr = opts.lr;
    lrvec = [];
    while length(lrvec) < opts.epoch
        lrvec = [lrvec, ones(1, opts.lrepoch)*cur_lr];
        cur_lr = cur_lr * opts.lrdecay;
    end
else
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
Htrain   = cnn_hash_test(net, batchFunc, imdb, train_id, opts);

test_id = find(imdb.images.set == 3);
Ytest   = imdb.images.labels(test_id)';
Htest   = cnn_hash_test(net, batchFunc, imdb, test_id, opts);

disp('Evaluating...');
opts.metric = 'mAP';
opts.unsupervised = false;
evaluate_deepMI(Htrain, Htest, Ytrain, Ytest, opts);
evaluate_fastap(Htrain, Htest, Ytrain, Ytest, opts);
%evaluate_opti(Htrain, Htest, Ytrain, Ytest, opts);
%evaluate_pess(Htrain, Htest, Ytrain, Ytest, opts);
end

% -------------------------------------------------------------------
% (not used) get solver
% -------------------------------------------------------------------
function h = get_solver(solverName)
switch solverName
    case 'sgd'
        h = [];
    case 'adam'
        h = @solver.adam;
    case 'rmsprop'
        h = @solver.rmsprop;
    case 'adagrad'
        h = @solver.adagrad;
    case 'adadelta'
        h = @solver.adadelta;
    otherwise
        error('unsupported opts.solver');
end
end
