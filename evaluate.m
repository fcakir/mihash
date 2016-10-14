function res = evaluate(Htrain, Htest, Ytrain, Ytest, opts, cateTrainTest)
% input: 
%   Htrain - (logical) training binary codes
%   Htest  - (logical) testing binary codes
%   Ytrain - (int32) training labels
%   Ytest  - (int32) testing labels
% output:
%  mAP - mean Average Precision
if nargin < 6, cateTrainTest = []; end
use_cateTrainTest = ~isempty(cateTrainTest);

trainsize = length(Ytrain);
testsize  = length(Ytest);

if strcmp(opts.metric, 'mAP')
    sim = single(2*Htrain-1)'*single(2*Htest-1);
    AP  = zeros(1, testsize);
    for j = 1:testsize
        labels = 2*double(Ytrain==Ytest(j))-1;
        [~, ~, info] = vl_pr(labels, double(sim(:, j)));
        AP(j) = info.ap;
    end
    AP = AP(~isnan(AP));
    res = mean(AP);
    myLogInfo(['mAP = ' num2str(res)]);

elseif ~isempty(strfind(opts.metric, 'mAP_'))
    % eval mAP on top N retrieved results
    assert(isfield(opts, 'mAP') & opts.mAP > 0);
    assert(opts.mAP < trainsize);
    N = opts.mAP;
    AP = zeros(1, testsize);
    sim = (2*single(Htrain)-1)'*(2*single(Htest)-1);

    % NOTE: parfor seems to run out of memory on Places
    for j = 1:testsize
        sim_j = sim(:, j);
        idx = [];
        for th = opts.nbits:-1:-opts.nbits
            idx = [idx; find(sim_j == th)];
            if length(idx) >= N, break; end
        end
        labels = 2*double(Ytrain(idx(1:N)) == Ytest(j)) - 1;
        [~, ~, info] = vl_pr(labels, double(sim_j(idx(1:N))));
        AP(j) = info.ap;
    end
    AP = AP(~isnan(AP));
    res = mean(AP);
    myLogInfo('mAP@(N=%d) = %g', N, res);

elseif ~isempty(strfind(opts.metric, 'prec_k'))
    % intended for PLACES, large scale
    K = opts.prec_k;
    prec_k = zeros(1, testsize);
    sim = single(2*Htrain-1)'*single(2*Htest-1);

    parfor i = 1:testsize
        labels = (Ytrain == Ytest(i));
        sim_i = sim(:, i);
        [~, I] = sort(sim_i, 'descend');
        I = I(1:K);
        prec_k(i) = mean(labels(I));
    end
    res = mean(prec_k);
    myLogInfo('Prec@(neighs=%d) = %g', K, res);


elseif ~isempty(strfind(opts.metric, 'prec_n'))
    N = opts.prec_n;
    R = opts.nbits;
    prec_n = zeros(1, testsize);
    sim = single(2*Htrain-1)'*single(2*Htest-1);

    % NOTE 'for' has better CPU usage
    for j=1:testsize
        if use_cateTrainTest
            labels = cateTrainTest(:, j);
        else
            labels = (Ytrain == Ytest(j));
        end
        ind = find(R-sim(:,j) <= 2*N);
        if ~isempty(ind)
            prec_n(j) = mean(labels(ind));
        end
    end
    res = mean(prec_n);
    myLogInfo('Prec@(radius=%d) = %g', N, res);

else
    error(['Evaluation metric ' opts.metric ' not implemented']);
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
