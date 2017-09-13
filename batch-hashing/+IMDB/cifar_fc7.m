function imdb = cifar_fc7(opts)
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
