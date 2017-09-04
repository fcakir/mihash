function res = evaluate(Htrain, Htest, Ytrain, Ytest, opts, Aff)
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
% input: 
%   Htrain - (logical) training binary codes
%   Htest  - (logical) testing binary codes
%   Ytrain - (int32) training labels
%   Ytest  - (int32) testing labels
% output:
%  mAP - mean Average Precision
if nargin < 6, Aff = []; end
hasAff = ~isempty(Aff);

if ~opts.unsupervised
    trainsize = length(Ytrain);
    testsize  = length(Ytest);
else
    [trainsize, testsize] = size(Aff);
end

if strcmp(opts.metric, 'mAP')
    sim = compare_hash_tables(Htrain, Htest);
    AP  = zeros(1, testsize);
    for j = 1:testsize
        labels = 2 * Aff(:, j) - 1;
        [~, ~, info] = vl_pr(labels, double(sim(:, j)));
        AP(j) = info.ap;
    end
    res = mean(AP(~isnan(AP)));
    logInfo(['mAP = ' num2str(res)]);

elseif ~isempty(strfind(opts.metric, 'mAP_'))
    % eval mAP on top N retrieved results
    assert(isfield(opts, 'mAP') & opts.mAP > 0);
    assert(opts.mAP < trainsize);
    N   = opts.mAP;
    sim = compare_hash_tables(Htrain, Htest);
    AP  = zeros(1, testsize);
    for j = 1:testsize
        sim_j = double(sim(:, j));
        idx = [];
        for th = opts.nbits:-1:-opts.nbits
            idx = [idx; find(sim_j == th)];
            if length(idx) >= N, break; end
        end
        idx = idx(1:N);
        labels = 2 * Aff(idx, j) - 1;
        [~, ~, info] = vl_pr(labels, sim_j(idx));
        AP(j) = info.ap;
    end
    res = mean(AP(~isnan(AP)));
    logInfo('mAP@(N=%d) = %g', N, res);

elseif ~isempty(strfind(opts.metric, 'prec_k'))
    % intended for PLACES, large scale
    K = opts.prec_k;
    sim = compare_hash_tables(Htrain, Htest);
    prec_k = zeros(1, testsize);
    for j = 1:testsize
        labels = Aff(:, j);
        [~, I] = sort(sim(:, j), 'descend');
        prec_k(i) = mean(labels(I(1:K)));
    end
    res = mean(prec_k);
    logInfo('Prec@(neighs=%d) = %g', K, res);

elseif ~isempty(strfind(opts.metric, 'prec_n'))
    N = opts.prec_n;
    R = opts.nbits;
    sim = compare_hash_tables(Htrain, Htest);
    prec_n = zeros(1, testsize);
    for j = 1:testsize
        labels = 2 * Aff(:, j) - 1;
        ind = find(R - sim(:,j) <= 2*N);
        if ~isempty(ind)
            prec_n(j) = mean(labels(ind));
        end
    end
    res = mean(prec_n);
    logInfo('Prec@(radius=%d) = %g', N, res);

else
    error(['Evaluation metric ' opts.metric ' not implemented']);
end
end

% ----------------------------------------------------------
function sim = compare_hash_tables(Htrain, Htest)
trainsize = size(Htrain, 2);
testsize  = size(Htest, 2);
if trainsize < 100e3
    sim = (2*single(Htrain)-1)'*(2*single(Htest)-1);
    sim = int8(sim);
else
    Ltest = 2*single(Htest)-1;
    sim = zeros(trainsize, testsize, 'int8');
    chunkSize = ceil(trainsize/10);
    for i = 1:ceil(trainsize/chunkSize)
        I = (i-1)*chunkSize+1 : min(i*chunkSize, trainsize);
        tmp = (2*single(Htrain(:,I))-1)' * Ltest;
        sim(I, :) = int8(tmp);
    end
    clear Ltest tmp
end
end


% ----------------------------------------------------------
function T = binsearch(x, k)
% x: input vector
% k: number of largest elements
% T: threshold
T = -Inf;
while numel(x) > k
    T0 = T;
    x0 = x;
    T  = mean(x);
    x  = x(x>T);
end
% for sanity
if numel(x) < k, T = T0; end
end
