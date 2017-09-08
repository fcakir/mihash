function DS = load_gist(opts, normalizeX)
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
% Load and prepare GIST features. The data paths must be changed. For all datasets,
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
%	thr_dist - (int)   For unlabelled datasets, corresponds to the distance 
%			   value to be used in determining whether two data instance
% 			   are neighbors. If their distance is smaller, then they are
% 			   considered neighbors.
%			   Given the standard setup, this threshold value
%			   is hard-wired to be compute from the 5th percentile 
% 			   distance value obtain through 2,000 training instance. 
% 			   See labelme case below.
%
if nargin < 2, normalizeX = 1; end
if ~normalizeX, logInfo('will NOT pre-normalize data'); end

tic;
load(fullfile(opts.dirs.data, 'LabelMe_GIST.mat'), 'gist');

% normalize features
if normalizeX
    X = bsxfun(@minus, gist, mean(gist,1));  % first center at 0
    X = normalize(double(X));  % then scale to unit length
end

ind = randperm(size(X, 1));
no_tst = 1000;

DS = [];
DS.Xtest  = X(ind(1:no_tst),:);
DS.Xtrain = X(ind(no_tst+1:end),:);
DS.Ytrain = [];
DS.Ytest  = [];
DS.thr_dist = prctile(pdist(Xtrain(1:2000,:), 'Euclidean'), 5); 

logInfo('[LabelMe_GIST] loaded in %.2f secs', toc);
end
