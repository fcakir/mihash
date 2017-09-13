function imdb = cifar(opts)
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

[data, labels, set, names] = cifar_load_images(opts);

imgSize = opts.imageSize;
if opts.normalize
    % NOTE: This normalization only applies when we're training on 32x32 images
    % directly. Do not do any normalization for imagenet pretrained VGG/Alexnet, 
    % for which resizing and mean subtraction are done on-the-fly during batch 
    % generation.
    assert(imgSize == 32);

    % normalize by image mean and std as suggested in `An Analysis of
    % Single-Layer Networks in Unsupervised Feature Learning` Adam
    % Coates, Honglak Lee, Andrew Y. Ng

    %if opts.contrastNormalization
        z = reshape(data,[],60000) ;
        z = bsxfun(@minus, z, mean(z,1)) ;
        n = std(z,0,1) ;
        z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
        data = reshape(z, imgSize, imgSize, 3, []) ;
    %end

    %if opts.whitenData
        z = reshape(data,[],60000) ;
        W = z(:,set == 1)*z(:,set == 1)'/60000 ;
        [V,D] = eig(W) ;
        % the scale is selected to approximately preserve the norm of W
        d2 = diag(D) ;
        en = sqrt(mean(d2)) ;
        z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
        data = reshape(z, imgSize, imgSize, 3, []) ;
    %end
    logInfo('Data normalized.');
end

imdb.images.data = data ;
imdb.images.labels = labels ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = names.label_names;
end



function [data, labels, set, clNames] = cifar_load_images(opts)
% Preapre the imdb structure, returns image data with mean image subtracted
unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
    {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 5), 3]);
if any(cellfun(@(fn) ~exist(fn, 'file'), files))
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
    fprintf('downloading %s\n', url) ;
    untar(url, opts.dataDir) ;
end

data   = cell(1, numel(files));
labels = cell(1, numel(files));
sets   = cell(1, numel(files));
for fi = 1:numel(files)
    fd = load(files{fi}) ;
    data{fi} = permute(reshape(fd.data',32,32,3,[]), [2 1 3 4]) ;
    labels{fi} = fd.labels' + 1;  % Index from 1
    sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));
labels = single(cat(2, labels{:})) ;

clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

end
