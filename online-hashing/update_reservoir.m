function [reservoir, update_ind] = update_reservoir(reservoir, ...
    points, labels, max_reservoir_size, W, unsupervised)
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
% 
% reservoir sampling, update step, based on random sort
% inputs:
%   reservoir: struct(X, Y, H, PQ, size)
% outputs:
%   update_ind: indices of updated entries ([] if no update)
%
if ~exist('unsupervised', 'var'), unsupervised = isempty(labels); end
n = size(points, 1);
if ~unsupervised, assert(n == size(labels, 1)); end;

if reservoir.size < max_reservoir_size
    % if reservoir not full, append (up to max_reservoir_size)
    n = min(n, max_reservoir_size - reservoir.size);
    reservoir.X = [reservoir.X; points(1:n, :)];
    if ~unsupervised
        reservoir.Y = [reservoir.Y; labels(1:n, :)];
    end
    reservoir.PQ = [reservoir.PQ; rand(n, 1)];
    update_ind = reservoir.size + (1:n);
else
    % full reservoir, update
    update_ind = [];
    for i = 1:n
        % pop max from priority queue
        [maxval, maxind] = max(reservoir.PQ);
        r = rand;
        if maxval > r
            % push into priority queue
            reservoir.PQ(maxind)   = r;
            reservoir.X(maxind, :) = points(i, :);
            if ~unsupervised
                reservoir.Y(maxind, :) = labels(i, :);
            end
            update_ind = [update_ind, maxind];
        end
    end
end
reservoir.size = size(reservoir.X, 1);

% if hash functions are given -- udpate entries
if exist('W', 'var')
    if isempty(reservoir.H)
        reservoir.H = (reservoir.X * W > 0);
    elseif ~isempty(update_ind)
        reservoir.H(update_ind, :) = (reservoir.X(update_ind, :) * W > 0);
    end
end
