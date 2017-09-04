function [obj, grad] = mutual_info(W_last, input, reservoir, no_bins, sigscale,...
    unsupervised, thr_dist, bool_gradient)
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
% additional information.
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
% Helper function for train_mihash.m									   
%
% INPUTS									   
%     W_last        - matrix, contains hash function parameters
%     input.X       - data point
%     input.Y       - label of X, empty if X has no labels
%     reservoir.X   - reservoir data points, reservoir_size x dimensionality
%     reservoir.Y   - reservoir labels corresponding to X, if empty then X does
%                     not have labels 
%     reservoir.H   - reservoir hash table, reservoir_size x nbits
%     reservoir.size- number of items in the reservoir
%     sigscale       - scaling parameter in sigmoid function
%     unsupervised  - 1 if neighborhood is defined using thr_dist see below
%     thr_dist      - numeric value for thresholding
%     bool_gradient - return gradient
%
% OUTPUTS
%     obj 	    - negative mutual information, see Eq. 7 in MIHash paper.
%     grad      - gradient matrix, see Eq. 11 in MIHash paper, each column
% 		      contains the gradients of a single hash function

if exist('unsupervised', 'var') == 0 
    unsupervised = false; 
elseif unsupervised
    assert(exist('thr_dist', 'var') == 1);
end
obj = []; grad = [];

X = input.X;
Y = input.Y;

if ~unsupervised     
    if size(reservoir.Y,2) > 1  % if multilabel
        Aff = (reservoir.Y * Y' > 0);    
    else  % if multiclass
        Aff = (reservoir.Y == Y);
    end
else
    Aff = pdist2(X, reservoir.X, 'euclidean') <= thr_dist;
end

% RELAXED hash codes to interval [-1, 1]
Hres = sigmoid(W_last'*reservoir.X', sigscale)';
nbits = size(Hres, 2);

% RELAXED hash input point X
hashX = sigmoid(W_last'*X', sigscale)'; % row vector, 1 x nbits

% compute distances from hash codes
hdist = (2*hashX - 1)*(2*Hres - 1)'; % row vector, 1 x reservoir_size
hdist = (-hdist + nbits)./2;   
assert(all(hdist >= 0) && all(hdist <= nbits));

% if Q is the (hamming) distance - x axis
% estimate P(Q|+), P(Q|-) & P(Q) manually
pQ = zeros(1, no_bins+1); 
deltaQ = nbits / no_bins;
bordersQ = 0:deltaQ:nbits;
assert(length(bordersQ) == no_bins+1);
for i=1:no_bins+1
    pQ(i) = sum(triPulse(bordersQ(i) - deltaQ, bordersQ(i) + deltaQ, hdist));
end
pQ = pQ ./ sum(pQ);
M = hdist(Aff); NM = hdist(~Aff);
pQCp = zeros(1, no_bins+1);
pQCn = zeros(1, no_bins+1);
for i=1:no_bins+1
    pQCp(i) = sum(triPulse(bordersQ(i) - deltaQ, bordersQ(i) + deltaQ, M));
    pQCn(i) = sum(triPulse(bordersQ(i) - deltaQ, bordersQ(i) + deltaQ, NM));
end

prCp = length(M) ./ (length(M) + length(NM));
prCn = 1 - prCp;
if sum(pQCp) ~= 0, pQCp = pQCp ./ sum(pQCp); end;
if sum(pQCn) ~= 0, pQCn = pQCn ./ sum(pQCn); end;
assert(sum(abs((pQCp*prCp+ pQCn*prCn) - pQ)) < 1e-6);

% estimate H(Q) entropy -> Qent
idx = find(pQ > 0);
Qent = -sum(pQ(idx).*log2(pQ(idx)));

% estimate H(Q|C) -> condent
idx = find(pQCp > 0);
p = -sum(pQCp(idx).*log2(pQCp(idx)));
idx = find(pQCn > 0);
n = -sum(pQCn(idx).*log2(pQCn(idx)));
condent = p * prCp + n * prCn;    

assert(Qent-condent >= 0);
obj = -(Qent - condent); % we're minimizing the mutual info.

Hres = Hres'; % nbits x reservoir_size
% Assumes hash codes are relaxed from {-1, 1} to [-1, 1]
if bool_gradient
    d_dh_phi = -0.5*Hres; % nbits x reservoir_size matrix: each column is --> \partial d_h(x, x^r) / \partial \Phi(x) = -\Phi(x^r) / 2
    d_delta_phi = zeros(nbits, reservoir.size, no_bins+1);
    d_pQCp_phi = zeros(no_bins+1, nbits);
    d_pQCn_phi = zeros(no_bins+1, nbits);
    for i=1:no_bins+1
        A = dTPulse(bordersQ(i) - deltaQ, bordersQ(i) + deltaQ, hdist);
        % each column of below matrix (RHS) --> [\partial \delta_{x^r,l} / \partial d_h(x, x^r)] x [\partial d_h(x, x^r) / \partial \Phi(x)] 
        % = \partial \delta_{x^r,l} / \partial \Phi(x)
        d_delta_phi(:,:,i) = bsxfun(@times, d_dh_phi, A); 
    end

    for i=1:no_bins+1
        % \partial p_{D,l}^+ / \partial \Phi(x)
        % having computed d_delta_phi, we just some the respective columns
        % that correspond to positive neighbors. 
        if length(M) ~= 0, d_pQCp_phi(i,:) = sum(d_delta_phi(:, Aff, i),2)'./length(M); end;%row vector
        % similar to above computation but for \partial p_{D,l}^- /
        % \partial Phi(x)
        if length(NM) ~= 0, d_pQCn_phi(i,:) = sum(d_delta_phi(:, ~Aff, i),2)'./length(NM); end; %row vector        
    end
    % \partial p_{D,l} / \partial \Phi(x)
    d_pQ_phi = d_pQCp_phi*prCp + d_pQCn_phi*prCn;
    t_log = ones(1, no_bins+1);
    idx = find(pQ > 0);
    t_log(idx) = t_log(idx) + log2(pQ(idx));

    % \partial H(D) / \Phi(x)
    d_H_phi = sum(bsxfun(@times, d_pQ_phi, t_log'), 1)'; % column vector, this is equal to negative gradient of entropy -grad H

    t_log_p = ones(1, no_bins+1);
    t_log_n = ones(1, no_bins+1);
    idx = find(pQCp > 0);
    idx2 = find(pQCn > 0);
    t_log_p(idx) = t_log_p(idx) + log2(pQCp(idx));
    t_log_n(idx2) = t_log_n(idx2) + log2(pQCn(idx2));

    % \partial H(D|C) / \partial \Phi(x)
    d_cond_phi = prCp * sum(bsxfun(@times, d_pQCp_phi, t_log_p'),1)' + ...
        prCn* sum(bsxfun(@times, d_pQCn_phi, t_log_n'), 1)'; % This is equal to negative gradient of cond entropy -grad H(|)

    % a vector
    d_MI_phi = d_H_phi - d_cond_phi; % This is equal to the gradient of negative MI, 

    % Since \Phi(x) = [\phi_1(x),...,\phi_b(x)] where \phi_i(x) = \sigma(w_i^t \times x)
    % take gradient of each \phi_i wrt to weight w_i, and multiply the
    % resulting vector with corresponding entry in d_MI_phi
    ty = sigscale * (W_last'*X'); % a vector
    grad = (bsxfun(@times, bsxfun(@times, repmat(X', 1, length(ty)), ...
        (sigmoid(ty, 1) .* (1 - sigmoid(ty, 1)) .* sigscale)'), d_MI_phi'));
end
end


function y = triPulse(a,c,x)
% a must be smaller than c
mid = (a+c)/2;
y = zeros(1, length(x));
ind = x > a & x <= mid;    
y(ind) = 1-abs(x(ind)-mid)./(c-mid);

ind = x > mid & x <= c;
y(ind) = 1-abs(x(ind)-mid)./(c-mid);
end


function y = dTPulse(a,c,x)
% a must be smaller than c
mid = (a+c)/2;
der = 2/(c-a);
y = zeros(1, length(x));
y(x > a & x <= mid) = der;
y(x > mid & x <= c) = -der;
end

