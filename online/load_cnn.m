function [Xtrain, Ytrain, Xtest, Ytest, Names] = load_cnn(opts, normalizeX)
% Load and prepare CNN features. The data paths must be changed. For all datasets,
% X represents the data matrix. Rows correspond to data instances and columns
% correspond to variables/features.
% Y represents the label matrix where each row corresponds to a label vector of 
% an item, i.e., for multiclass datasets this vector has a single dimension and 
% for multilabel datasets the number of columns of Y equal the number of labels
% in the dataset. Y can be empty for unsupervised datasets.
% 
%
% INPUTS
%	opts   - (struct)  Parameter structure.
%   normalizeX - (int)     Choices are {0, 1}. If normalizeX = 1, the data is 
% 			   mean centered and unit-normalized. 
% 		
% OUTPUTS
% 	Xtrain - (nxd) 	   Training data matrix, each row corresponds to a data
%			   instance.
%	Ytrain - (nxl)     Training data label matrix. l=1 for multiclass datasets.
%			   For unsupervised dataset Ytrain=[], see LabelMe in 
%			   load_gist.m
%	Xtest  - (nxd)     Test data matrix, each row corresponds to a data instance.
%	Ytest  - (nxl)	   Test data label matrix, l=1 for multiclass datasets. 
%			   For unsupervised dataset Ytrain=[], see LabelMe in 
%			   load_gist.m
%	Names  - (struct)  For future release.
%
% 
if nargin < 2, normalizeX = 1; end
if ~normalizeX, logInfo('will NOT pre-normalize data'); end

% NOTE: labels are originally [0, L-1], first add 1 to make [1, L]
%       then multiply by 10 to make [10, L*10]
%
%       Next, for each item, if HIDE label in training, +1 to its Y
%       So eg. for first class, labeled ones have 10, unlabeled have 11
%
%       At test time labels can be recovered by dividing 10

tic;
if strcmp(opts.dataset, 'cifar')
    basedir = '/research/codebooks/hashing_project/data';
    load([basedir '/cifar-10/descriptors/trainCNN.mat']); % trainCNN
    load([basedir '/cifar-10/descriptors/traininglabelsCNN.mat']); % traininglabels
    load([basedir '/cifar-10/descriptors/testCNN.mat']); % testCNN
    load([basedir '/cifar-10/descriptors/testlabelsCNN.mat']); % testlabels
    X = [trainCNN; testCNN];
    Y = [traininglabels+1; testlabels+1];
    ind = randperm(length(Y));
    X = X(ind, :);
    Y = Y(ind);
    clear ind
    Y = Y .* 10;
    T = 100;

    if normalizeX 
        % normalize features
        X = bsxfun(@minus, X, mean(X,1));  % first center at 0
        X = normalize(double(X));  % then scale to unit length
    end

    % fully supervised
    % TODO names
    [ind_train, ind_test, Ytrain, Ytest] = split_train_test(X, Y, T, 0);
    Xtrain = X(ind_train, :);
    Xtest  = X(ind_test, :);
    
    if opts.val_size > 0
	assert(length(Ytrain) >=  opts.val_size);
    	logInfo('Doing validation!');
	ind_ = randperm(length(Ytrain));
	Xtrain = Xtrain(ind_(1:opts.val_size), :);
	Ytrain = Ytrain(ind_(1:opts.val_size), :);
	ind_train = ind_train(ind_(1:opts.val_size));
    end

    clear Names
    Names.train = num2cell(ind_train);
    Names.test = num2cell(ind_test);


elseif strcmp(opts.dataset, 'sun')

    load('/research/codebooks/hashing_project/data/sun397/alltrain_sun_fc7_final.mat');

    X = alldata;
    Y = allclasses;
    Y = (Y + 1)*10;  % NOTE labels are 0 to 396
    T = 10;
    clear alldata allclasses

    if normalizeX 
        % normalize features
        X = bsxfun(@minus, X, mean(X,1));  % first center at 0
        X = normalize(double(X));  % then scale to unit length
    end
    [ind_train, ind_test, Ytrain, Ytest] = ...
        split_train_test(X, Y, T ,0);

    Xtrain = X(ind_train, :);
    Xtest  = X(ind_test, :);

    if opts.val_size > 0
	assert(length(Ytrain) >=  opts.val_size);
    	logInfo('Doing validation!');
	ind_ = randperm(length(Ytrain));
	Xtrain = Xtrain(ind_(1:opts.val_size), :);
	Ytrain = Ytrain(ind_(1:opts.val_size), :);
	ind_train = ind_train(ind_(1:opts.val_size));
    end
	Names = [];
elseif strcmp(opts.dataset, 'places')
    basedir = '/research/object_detection/data';
    % loads variables: pca_feats, labels, images
    clear pca_feats labels images
    load([basedir '/places/places_alexnet_fc7pca128.mat']);
    X = pca_feats;
    Y = (labels + 1)*10;
    T = 20;
    L = opts.labelsPerCls;  % default 2500, range {0}U[500, 5000]

    if normalizeX 
        % normalize features
        X = bsxfun(@minus, X, mean(X,1));  % first center at 0
        X = normalize(double(X));  % then scale to unit length
    end

    % semi-supervised
    [ind_train, ind_test, Ytrain, Ytest] = split_train_test(X, Y, T, L);
    Xtrain = X(ind_train, :);
    Xtest  = X(ind_test, :);
    
    if opts.val_size > 0
	assert(length(Ytrain) >=  opts.val_size);
    	logInfo('Doing validation!');
	ind_ = randperm(length(Ytrain));
	Xtrain = Xtrain(ind_(1:opts.val_size), :);
	Ytrain = Ytrain(ind_(1:opts.val_size), :);
	ind_train = ind_train(ind_(1:opts.val_size));
    end

    clear Names
    Names.train = images(ind_train);
    Names.test  = images(ind_test);


elseif strcmp(opts.dataset, 'nus')
    basedir = '/research/codebooks/hashing_project/data';
    load([basedir '/nuswide/AllNuswide_fc7.mat']);  % FVs
    Y = load([basedir '/nuswide/AllLabels81.txt']);
    use21FrequentConcepts = 1;
    if use21FrequentConcepts
    	logInfo('Using 21 most frequent concepts, removing rest...');
	[~, fi_] = sort(sum(Y, 1), 'descend');
	Y(:, fi_(22:end)) = [];
	fi2_ = find(sum(Y, 2) == 0);
	Y(fi2_,:) = [];
	FVs(fi2_,:) = [];
	logInfo('No. of points=%g, dimensionality=%g, No. of labels=%g', ...
		size(FVs,1), size(FVs, 2), size(Y, 2));
    end
    X = double(FVs);  clear FVs
    T = 100;
    if normalizeX 
        % normalize features
        X = bsxfun(@minus, X, mean(X,1));  % first center at 0
        X = normalize(double(X));  % then scale to unit length
    end

    % TODO Names
    [Xtrain, Ytrain, Xtest, Ytest] = split_train_test_nus(X, Y, T, ...
    	use21FrequentConcepts);
    Names = [];
    if opts.val_size > 0
	assert(size(Ytrain, 1) >=  opts.val_size);
    	logInfo('Doing validation!');
	ind_ = randperm(length(Ytrain));
	Xtrain = Xtrain(ind_(1:opts.val_size), :);
	Ytrain = Ytrain(ind_(1:opts.val_size), :);
    end

else, error(['unknown dataset: ' opts.dataset]); end

whos Xtrain Ytrain Xtest Ytest
logInfo('Dataset "%s" loaded in %.2f secs', opts.dataset, toc);
end

% --------------------------------------------------------
function [ind_train, ind_test, Ytrain, Ytest] = split_train_test(X, Y, T, L)
% X: original features
% Y: original labels
% T: # test points per class
% L: [optional] # labels to retain per class
if nargin < 4, L = 0; end

% randomize
%I = randperm(size(X, 1));
%X = X(I, :);
%Y = Y(I);

D = size(X, 2);

labels = unique(Y);
ntest  = length(labels) * T;
ntrain = size(X, 1) - ntest;
%Xtrain = zeros(ntrain, D);  Xtest = zeros(ntest, D);
Ytrain = zeros(ntrain, 1);  Ytest = zeros(ntest, 1);
ind_train = [];
ind_test  = [];

% construct test and training set
cnt = 0;
for i = 1:length(labels)
    % find examples in this class, randomize ordering
    ind = find(Y == labels(i));
    n_i = length(ind);
    ind = ind(randperm(n_i));

    % assign test
    %Xtest((i-1)*T+1:i*T, :) = X(ind(1:T), :);
    Ytest((i-1)*T+1:i*T)    = labels(i);
    ind_test = [ind_test; ind(1:T)];

    % assign train
    st = cnt + 1; 
    ed = cnt + n_i - T;
    %Xtrain(st:ed, :) = X(ind(T+1:end), :);
    ind_train = [ind_train; ind(T+1:end)];
    Ytrain(st:ed)    = labels(i);
    if L > 0  
        % if requested, hide some labels
        if st + L > ed
            warning(sprintf('%s Class%d: ntrain=%d<%d=labelsPerCls, keeping all', ...
                labels(i), n_i-T, L));
        else
            % add 1 to mark unlabeled items
            Ytrain(st+L: ed) = Ytrain(st+L: ed) + 1;
        end
    end
    cnt = ed;
end

% randomize again
ind    = randperm(ntrain);
%Xtrain = Xtrain(ind, :);
Ytrain = Ytrain(ind);
ind_train = ind_train(ind);

ind    = randperm(ntest);
%Xtest  = Xtest(ind, :);
Ytest  = Ytest(ind);
ind_test = ind_test(ind);
end


% --------------------------------------------------------
function [Xtrain, Ytrain, Xtest, Ytest] = ...
    split_train_test_nus(gist, tags, tstperclass, ufq)

% normalize features
gist = bsxfun(@minus, gist, mean(gist,1));  % first center at 0
gist = normalize(gist);  % then scale to unit length

% construct test and training set
num_classes = size(tags, 2);
%if ufq == 0, num_classes = 81;else, num_classes = 21; end;
testsize    = num_classes * tstperclass;
ind         = randperm(size(gist, 1));
Xtest       = gist(ind(1:testsize), :);
Ytest       = tags(ind(1:testsize), :);
Xtrain      = gist(ind(testsize+1:end), :);
Ytrain      = tags(ind(testsize+1:end), :);

% randomize again
ind    = randperm(size(Xtrain, 1));
Xtrain = Xtrain(ind, :);
Ytrain = Ytrain(ind, :);
ind    = randperm(size(Xtest, 1));
Xtest  = Xtest(ind, :);
Ytest  = Ytest(ind, :);
end
