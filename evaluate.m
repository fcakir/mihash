function res = evaluate(Htrain, Htest, opts, Aff)
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
% Given the hash codes of the training data (Htrain) and the hash codes of the test
% data (Htest) evaluates the performance.
% 
% INPUTS
% 	Htrain - (logical) Matrix containing the hash codes of the training
% 			   data. Each column corresponds to a hash code. 
%  	Htest  - (logical) Matrix containing the hash codes of the test data. 
%			   Each column corresponds to a hash code.
%	opts   - (struct)  Parameter structure.
%       Aff    - (logical) Neighbor indicator matrix. trainingsize x testsize. 
%
% OUTPUTS
%  	res    - (float) performance value as determined by opts.metric

[trainsize, testsize] = size(Aff);

if strcmp(opts.metric, 'mAP')
    % eval mAP
    AP  = zeros(1, testsize);
    sim = compare_hash_tables(Htrain, Htest);

    for j = 1:testsize
        labels = 2*Aff(:, j)-1;
        [~, ~, info] = vl_pr(labels, double(sim(:, j)));
        AP(j) = info.ap;
    end
    res = mean(AP(~isnan(AP)));
    logInfo(['mAP = ' num2str(res)]);

elseif ~isempty(strfind(opts.metric, 'mAP_'))
    % eval mAP on top N retrieved results
    assert(isfield(opts, 'mAP') & opts.mAP > 0);
    assert(opts.mAP < trainsize);
    N = opts.mAP; 
    AP = zeros(1, testsize);
    sim = compare_hash_tables(Htrain, Htest);

    for j = 1:testsize
        sim_j = double(sim(:, j));
        idx = [];
        for th = opts.nbits:-1:-opts.nbits
            idx = [idx; find(sim_j == th)];
            if length(idx) >= N, break; end
        end
        labels = 2*Aff(idx(1:N), j)-1;
        [~, ~, info] = vl_pr(labels, sim_j(idx(1:N)));
        AP(j) = info.ap;
    end
    res = mean(AP(~isnan(AP)));
    logInfo('mAP@(N=%d) = %g', N, res);

elseif ~isempty(strfind(opts.metric, 'prec_k'))
    % eval precision @ k (nearest neighbors)
    K = opts.prec_k; 
    prec_k = zeros(1, testsize);
    sim = compare_hash_tables(Htrain, Htest);

    for i = 1:testsize
        labels = Aff(:, i);
        sim_i = sim(:, i);
        [~, I] = sort(sim_i, 'descend');
        I = I(1:K);
        prec_k(i) = mean(labels(I));
    end
    res = mean(prec_k);
    logInfo('Prec@(neighs=%d) = %g', K, res);

elseif ~isempty(strfind(opts.metric, 'prec_n'))
    % eval precision @ N (Hamming ball radius)
    N = opts.prec_n; 
    R = opts.nbits;
    prec_n = zeros(1, testsize);
    sim = compare_hash_tables(Htrain, Htest);

    for j=1:testsize
        labels = Aff(:, j);
        ind = find(R-sim(:,j) <= 2*N);
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
trainsize = size(Htrain, 1);
testsize  = size(Htest, 1);
if trainsize < 100e3
    sim = (2*single(Htrain)-1)*(2*single(Htest)-1)';
    sim = int8(sim);
else
    % for large scale data: process in chunks
    Ltest = 2*single(Htest)-1;
    sim = zeros(trainsize, testsize, 'int8');
    chunkSize = ceil(trainsize/10);
    for i = 1:ceil(trainsize/chunkSize)
        I = (i-1)*chunkSize+1 : min(i*chunkSize, trainsize);
        tmp = (2*single(Htrain(:,I))-1) * Ltest';
        sim(I, :) = int8(tmp);
    end
    clear Ltest tmp
end
end
