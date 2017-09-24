function update_table = trigger_update(iter, W_last, W, reservoir, ...
    Hres_new, opts)
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

update_table = false;

% ----------------------------------------------
% no update if hash mapping has not changed
if sum(abs(W_last(:) - W(:))) < 1e-6
    update_table = false;
    logInfo('[W no change] iter %d/%d, update = 0', iter, opts.num_iters);
    return;
end

% ----------------------------------------------
% at an update interval

if opts.reservoirSize <= 0 || strcmp(opts.trigger, 'fix')
    % no reservoir or 'fix' -- update
    update_table = true;
    logInfo('[Fix] iter %d/%d, update = 1', iter, opts.num_iters);

elseif opts.reservoirSize > 0 && opts.updateInterval > 0
    % using reservoir + MI criterion
    % signal update of hash table, if MI improvement > threshold
    assert(strcmp(opts.trigger, 'mi'));

    % affinity matrix
    Aff = affinity(reservoir.X, reservoir.X, reservoir.Y, reservoir.Y, opts);

    % MI improvement
    mi_old  = eval_mutualinfo(reservoir.H, Aff);
    mi_new  = eval_mutualinfo(Hres_new, Aff);
    mi_impr = mi_new - mi_old;

    % update?
    update_table = mi_impr > opts.triggerThresh;
    logInfo('[MI] iter %d/%d, impr = %g, update = %d', iter, opts.num_iters, ...
        mi_impr, update_table);
end

end


% --------------------------------------------------------------------
function mi = eval_mutualinfo(H, affinity)
% distance
[num, nbits] = size(H);
hdist = (2*H - 1) * (2*H - 1)';
hdist = (nbits - hdist) / 2;   

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
