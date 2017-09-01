function imdb = cifar_fc7(opts)
% load precomputed CNN features
if ~isempty(strfind(computer, 'WIN'))
    basedir = '\\ivcfs1\codebooks\hashing_project\data';
else
    basedir = '/research/codebooks/hashing_project/data';
end
% TODO put into a single file and give download link
load([basedir '/cifar-10/descriptors/trainCNN.mat']); % trainCNN
load([basedir '/cifar-10/descriptors/traininglabelsCNN.mat']); % traininglabels
load([basedir '/cifar-10/descriptors/testCNN.mat']); % testCNN
load([basedir '/cifar-10/descriptors/testlabelsCNN.mat']); % testlabels

data = [testCNN; trainCNN];
labels = [testlabels; traininglabels] + 1;
sets = split_cifar(labels, opts);

% remove mean in any case
Xtrain = data(sets==1, :);
dataMean = mean(Xtrain, 1);
data = bsxfun(@minus, data, dataMean);

if opts.normalize
    % unit-length
    rownorm = sqrt(sum(data.^2, 2));
    data = bsxfun(@rdivide, data, rownorm);
end

imdb.images.data = permute(single(data), [3 4 2 1]);
imdb.images.labels = single(labels');
imdb.images.set = uint8(sets');
imdb.meta.sets = {'train', 'val', 'test'} ;
end
