function [update_table, ret_val] = trigger_update(iter, opts, ...
    W_last, W, reservoir, Hres_new, varargin)
% Copyright (c) 2017, Fatih Cakir, Kun He, Saral Adel Bargal, Stan Sclaroff 
% All rights reserved.
% 
% If used for please cite the below paper:
%
% "MIHash: Online Hashing with Mutual Information", 
% Fatih Cakir*, Kun He*, Sarah Adel Bargal, Stan Sclaroff
% (* equal contribution)
% International Conference on Computer Vision (ICCV) 2017
% 
% Usage of code from authors not listed above might be subject
% to different licensing. Please check with the corresponding authors for
% information.
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

update_table = false;
ret_val = -1;

% ----------------------------------------------
% update on first iteration
if (iter == 1)
    update_table = true;  
    return;
end

% ----------------------------------------------
% no update if hash mapping has not changed
if sum(abs(W_last(:) - W(:))) < 1e-6
    update_table = false;
    return;
end

% ----------------------------------------------
% no reservoir -- use updateInterval
if opts.reservoirSize <= 0 || strcmp(opts.trigger, 'fix')
    update_table = ~mod(iter*opts.batchSize, opts.updateInterval);
    return;
end

% ----------------------------------------------
% using reservoir + MI criterion
% signal update of hash table, when:
%   1) we're at an updateInterval, AND
%   2) MI improvement > threshold
if opts.updateInterval > 0  &&  ...
        mod(iter*opts.batchSize, opts.updateInterval) == 0

    assert(strcmp(opts.trigger, 'mi'));
    mi_impr = trigger_mutualinfo(iter, W, W_last, reservoir.X, reservoir.Y, ...
        reservoir.H, Hres_new, reservoir.size, varargin{:});
    update_table = mi_impr > opts.miThresh;
    logInfo('MI improvement = %g, update = %d', mi_impr, update_table);
    ret_val = mi_impr;
end

end


% -------------------------------------------------------------------------
function mi_impr = trigger_mutualinfo(iter, W, W_last, X, Y, ...
    Hres, Hnew, reservoir_size, unsupervised, thr_dist)

nbits = size(Hres, 2);
assert(nbits == size(Hnew,2));
assert(isequal(reservoir_size, size(Hres,1), size(Hnew,1)));
assert(isequal((W_last'*X' > 0)', Hres));
assert((~unsupervised && ~isempty(Y)) || (unsupervised && isempty(Y)));

% affinity matrix
if exist('unsupervised', 'var') == 0 
    unsupervised = false; 
    Aff = (repmat(Y,1,length(Y)) == repmat(Y,1,length(Y))');
elseif unsupervised
    assert(exist('thr_dist', 'var') == 1);
    Aff = squareform(pdist(X, 'euclidean')) <= thr_dist;
end

mi_old  = eval_mutualinfo(Hres, Aff);
mi_new  = eval_mutualinfo(Hnew, Aff);
mi_impr = mi_new - mi_old;
end


% --------------------------------------------------------------------
function mi = eval_mutualinfo(H, affinity)
% distance
num   = size(H, 1);
nbits = size(H, 2);
hdist = (2*H - 1) * (2*H - 1)';
hdist = (-hdist + nbits)./2;   

% let Q be the Hamming distance
% estimate P(Q|+), P(Q|-) & P(Q)
condent = zeros(1, num);
Qent = zeros(1, num);
for j = 1:num
    D  = hdist(j, :); 
    M  = D( affinity(j, :)); 
    NM = D(~affinity(j, :));
    prob_Q_Cp = histcounts(M,  0:1:nbits);  % raw P(Q|+)
    prob_Q_Cn = histcounts(NM, 0:1:nbits);  % raw P(Q|-)
    sum_Q_Cp  = sum(prob_Q_Cp);
    sum_Q_Cn  = sum(prob_Q_Cn);
    prob_Q    = (prob_Q_Cp + prob_Q_Cn)/(sum_Q_Cp + sum_Q_Cn);
    prob_Q_Cp = prob_Q_Cp/sum_Q_Cp;
    prob_Q_Cn = prob_Q_Cn/sum_Q_Cn;
    prob_Cp   = length(M)/(length(M) + length(NM)); %P(+)
    prob_Cn   = 1 - prob_Cp; % P(-) 

    % estimate H(Q) entropy
    idx = find(prob_Q > 0);
    Qent(j) = -sum(prob_Q(idx).*log2(prob_Q(idx)));

    % estimate H(Q|C)
    idx = find(prob_Q_Cp > 0);
    p   = -sum(prob_Q_Cp(idx).*log2(prob_Q_Cp(idx)));
    idx = find(prob_Q_Cn > 0);
    n   = -sum(prob_Q_Cn(idx).*log2(prob_Q_Cn(idx)));
    condent(j) = p * prob_Cp + n * prob_Cn;    
end

mi = Qent - condent;
mi(mi < 0) = 0;  % deal with numerical inaccuracies
mi = mean(mi);
end
