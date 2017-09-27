function imdb = cifar_fc7(opts)

load([opts.dataDir '/CIFAR10_VGG16_fc7.mat']);

data = [testCNN; trainCNN];
labels = [testLabels; trainLabels] + 1;
sets = IMDB.split_cifar(labels, opts);

% remove mean in any case
Xtrain = data(sets==1, :);
dataMean = mean(Xtrain, 1);
data = bsxfun(@minus, data, dataMean);

if opts.normalize
    rownorm = sqrt(sum(data.^2, 2));
    data = bsxfun(@rdivide, data, rownorm);
    logInfo('Data normalized.');
end

imdb.images.data = permute(single(data), [3 4 2 1]);
imdb.images.labels = single(labels');
imdb.images.set = uint8(sets');
imdb.meta.sets = {'train', 'val', 'test'} ;
end
