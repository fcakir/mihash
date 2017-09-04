function [top] = mi_forward(layer, bot, top)
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
Y = squeeze(layer.class); % Nx1
X = squeeze(bot.x);  % 1x1xBxN -> BxN, raw scores (logits) for each bit
[nbits, N] = size(X);

opts = layer.opts;
onGPU = numel(opts.gpus) > 0;
if ~opts.unsupervised, assert(size(Y, 1) == N); end

% histogram bin centers & width
no_bins = opts.nbins;
deltaD = nbits / no_bins;
centersD = 0: deltaD: nbits;

% get NxN affinity matrix
if opts.unsupervised     
    X_raw = squeeze(layer.rawinput)';    
    Aff = squareform(pdist(X_raw, 'euclidean')) <= opts.thr_dist;
else
    if size(Y, 2) > 1  % if multilabel
        Aff = (Y * Y' > 0);    
    else  % if multiclass
        Aff = bsxfun(@eq, Y', Y);
    end
end
Xp = logical(Aff - diag(diag(Aff)));
Xn = ~Aff;
if onGPU
    Xp = gpuArray(Xp);
    Xn = gpuArray(Xn);
end

% compute distances from hash codes
phi = 2*sigmoid(X, opts.sigscale) - 1;  % RELAXED hash codes to interval [-1, 1]
hdist = phi' * phi;  % NxN pairwise dist matrix
hdist = (-hdist + nbits)/2;   

% estimate discrete distributions
prCp = sum(Xp, 2) ./ (N-1);
prCn = 1 - prCp;
pDCp = zeros(N, no_bins+1);
pDCn = zeros(N, no_bins+1);
if onGPU
    pDCp = gpuArray(pDCp);
    pDCn = gpuArray(pDCn);
end

% new version, better when L<N
for l = 1:no_bins+1
    pulse = triPulse(hdist, centersD(l), deltaD);  % NxN
    pDCp(:, l) = sum(pulse .* Xp, 2);
    pDCn(:, l) = sum(pulse .* Xn, 2);
end

% pD
pD = (pDCp + pDCn) ./ (N-1);
% normalize
sum_pDCp = sum(pDCp, 2);  nz_p = sum_pDCp > 0;
sum_pDCn = sum(pDCn, 2);  nz_n = sum_pDCn > 0;
pDCp(nz_p, :) = bsxfun(@rdivide, pDCp(nz_p, :), sum_pDCp(nz_p));
pDCn(nz_n, :) = bsxfun(@rdivide, pDCn(nz_n, :), sum_pDCn(nz_n));

% compute entropies
y0 = zeros(size(pD));
if onGPU, y0 = gpuArray(y0); end
ent_D   = ent(pD, y0);  % H(D)
ent_D_C = prCp .* ent(pDCp, y0) + prCn .* ent(pDCn, y0);  % H(D|C)

% loss
top.x = -sum(single(ent_D - ent_D_C));  % maximize MI -> minimize -MI
top.aux = [];
top.aux.phi = phi;
top.aux.Xp = Xp;
top.aux.Xn = Xn;
top.aux.distance = hdist;
top.aux.prCp = prCp;
top.aux.prCn = prCn;
top.aux.pDCp = pDCp;
top.aux.pDCn = pDCn;
top.aux.pD   = pD;
end


function y = sigmoid(x, p)
y = p(1) * x - p(2);
y(y>20) = 20;
y = 1 ./ (1 + exp(-y));
end


function y = triPulse(D, mid, delta)
% triPulse: triangular pulse
%
%     D: input matrix of distance values
%   mid: scalar, the center of some histogram bin
% delta: scalar, histogram bin width
%
% For histogram bin mid, compute the contribution y ("pulse") 
% from every element in D.  
% Interpolation using the triangular kernel
ind = (mid-delta < D) & (D <= mid+delta);
y   = 1 - abs(D - mid) / delta;
y   = y .* ind;
end


function H = ent(p, y0)
logp = y0;
logp(p>0) = log2(p(p>0));
H = -sum(p .* logp, 2);
end
