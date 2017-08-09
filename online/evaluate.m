function res = evaluate(Htrain, Htest, Ytrain, Ytest, opts, Aff)
% Given the hash codes of the training data (Htrain) and the hash codes of the test
% data (Htest) evaluates the performance.
% 
% INPUTS
% 	Htrain - (logical) Matrix containing the hash codes of the training
% 			   data. Each column corresponds to a hash code. 
%  	Htest  - (logical) Matrix containing the hash codes of the test data. 
%			   Each column corresponds to a hash code.
%  	Ytrain - (int) 	   Training data labels. For multilabel datasets such as nuswide
% 			   each column corresponds to a label. For unsupervised 
% 			   datasets such as LabelMe, Ytrain is set to [].
%   	Ytest  - (int) 	   Testing data labels. 
%	opts   - (struct)  Parameter structure.
%       Aff    - (logical) Neighbor indicator matrix. trainingsize x testsize. 
%			   May be empty to save memory. Then the indicator matrix 
% 			   is computer on-the-fly, see below. 
%
% OUTPUTS
%  	res    - (float) performance value as determined by opts.metric, e.g., mAP value.

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

    ncpu = feature('numcores');
    set_parpool(min(round(ncpu/2), 8));
    % if hasAff
    % TODO remove vlfeat
    for j = 1:testsize
        labels = 2*Aff(:, j)-1;
        [~, ~, info] = vl_pr(labels, double(sim(:, j)));
        AP(j) = info.ap;
    end
    % else
    %     for j = 1:testsize
    %         labels = 2*double(Ytrain==Ytest(j))-1;
    %         [~, ~, info] = vl_pr(labels, double(sim(:, j)));
    %         AP(j) = info.ap;
    %     end
    % end
    AP = AP(~isnan(AP));
    res = mean(AP);
    logInfo(['mAP = ' num2str(res)]);

elseif ~isempty(strfind(opts.metric, 'mAP_'))
    % eval mAP on top N retrieved results
    assert(isfield(opts, 'mAP') & opts.mAP > 0);
    assert(opts.mAP < trainsize);
    N = opts.mAP; 
    AP = zeros(1, testsize);
    sim = compare_hash_tables(Htrain, Htest);

    ncpu = feature('numcores');
    set_parpool(min(round(ncpu/2), 8));
    % if hasAff
    % TODO remove vlfeat
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
    % else
    %     for j = 1:testsize
    %         sim_j = double(sim(:, j));
    %         idx = [];
    %         for th = opts.nbits:-1:-opts.nbits
    %             idx = [idx; find(sim_j == th)];
    %             if length(idx) >= N, break; end
    %         end
    %         labels = 2*double(Ytrain(idx(1:N)) == Ytest(j)) - 1;
    %         [~, ~, info] = vl_pr(labels, sim_j(idx(1:N)));
    %         AP(j) = info.ap;
    %     end
    % end
    AP = AP(~isnan(AP));
    res = mean(AP);
    logInfo('mAP@(N=%d) = %g', N, res);

elseif ~isempty(strfind(opts.metric, 'prec_k'))
    % intended for PLACES, large scale
    K = opts.prec_k; 
    prec_k = zeros(1, testsize);
    sim = compare_hash_tables(Htrain, Htest);

    ncpu = feature('numcores');
    set_parpool(round(ncpu/2));
    for i = 1:testsize
       %  if hasAff
        labels = Aff(:, i);
        % else
        %     labels = (Ytrain == Ytest(i));
	% end
        sim_i = sim(:, i);
        [~, I] = sort(sim_i, 'descend');
        I = I(1:K);
        prec_k(i) = mean(labels(I));
    end
    res = mean(prec_k);
    logInfo('Prec@(neighs=%d) = %g', K, res);


elseif ~isempty(strfind(opts.metric, 'prec_n'))
    N = opts.prec_n; 
    R = opts.nbits;
    prec_n = zeros(1, testsize);
    sim = compare_hash_tables(Htrain, Htest);

    for j=1:testsize
        % if hasAff
        labels = Aff(:, j);
       %  else
       %      labels = (Ytrain == Ytest(j));
       %  end
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
