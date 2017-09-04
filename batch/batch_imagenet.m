function [images, labels] = batch_imagenet(imdb, batch, imgSize, meanImage)
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

% get images
if ~iscell(imdb.images.data)
    % already loaded in imdb
    images = imdb.images.data(:, :, :, batch) ;
    % normalization
    if imgSize ~= size(images, 1)
        images = imresize(images, [imgSize, imgSize]);
    end
    images = bsxfun(@minus, images, meanImage);
    % get labels
    if isempty(imdb.images.labels)
        itrain = find(imdb.images.set == 1);
        [~, labels] = ismember(batch, itrain);
    else
        labels = permute(imdb.images.labels(:, batch), [3 4 2 1]);
    end
else
    % train or test? train: use data augmentation
    args = {'Gpu', 'Pack', ...
            'NumThreads', 4, ...
            'Resize', [imgSize imgSize], ...
            'Interpolation', 'bicubic', ...
            'subtractAverage', meanImage};
    if imdb.images.set(batch(1)) == 1
        args{end+1} = 'Flip';
    end
    % imdb.images.data is a cell array of filepaths
    % first call: prefetch
    vl_imreadjpeg(imdb.images.data(batch), args{:}, 'prefetch');
    % get labels now
    if isempty(imdb.images.labels)
        itrain = find(imdb.images.set == 1);
        [~, labels] = ismember(batch, itrain);
    else
        labels = permute(imdb.images.labels(:, batch), [3 4 2 1]);
    end
    % second call to actually get images
    images = vl_imreadjpeg(imdb.images.data(batch), args{:});
    images = images{1};
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function data = getImageBatch(imagePaths, varargin)
% GETIMAGEBATCH  Load and jitter a batch of images

opts.useGpu = false ;
opts.prefetch = false ;
opts.numThreads = 1 ;

opts.imageSize = [227, 227] ;
opts.cropSize = 227 / 256 ;
opts.keepAspect = true ;
opts.subtractAverage = [] ;

opts.jitterFlip = false ;
opts.jitterLocation = false ;
opts.jitterAspect = 1 ;
opts.jitterScale = 1 ;
opts.jitterBrightness = 0 ;
opts.jitterContrast = 0 ;
opts.jitterSaturation = 0 ;

opts = vl_argparse(opts, varargin);

args{1} = {imagePaths, ...
           'NumThreads', opts.numThreads, ...
           'Pack', ...
           'Interpolation', 'bicubic', ...
           'Resize', opts.imageSize(1:2), ...
           'CropSize', opts.cropSize * opts.jitterScale, ...
           'CropAnisotropy', opts.jitterAspect, ...
           'Brightness', opts.jitterBrightness, ...
           'Contrast', opts.jitterContrast, ...
           'Saturation', opts.jitterSaturation} ;

if ~opts.keepAspect
  % Squashign effect
  args{end+1} = {'CropAnisotropy', 0} ;
end

if opts.jitterFlip
  args{end+1} = {'Flip'} ;
end

if opts.jitterLocation
  args{end+1} = {'CropLocation', 'random'} ;
else
  args{end+1} = {'CropLocation', 'center'} ;
end

if opts.useGpu
  args{end+1} = {'Gpu'} ;
end

if ~isempty(opts.subtractAverage)
  args{end+1} = {'SubtractAverage', opts.subtractAverage} ;
end

args = horzcat(args{:}) ;

if opts.prefetch
  vl_imreadjpeg(args{:}, 'prefetch') ;
  data = [] ;
else
  data = vl_imreadjpeg(args{:}) ;
  data = data{1} ;
end
end
