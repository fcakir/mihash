function evaluation = Evaluate_Fast(nBase, nQuery, linkGt, loopbits, codeDir, rsltDir)

tic;

addpath('./Evaluation.Tools/');

if ~isempty(nBase) && ~isempty(nQuery)
    instCntBase = sum(nBase);
    instCntQuery = sum(nQuery);
    queryIndBeg = 1;
    if iscolumn(nBase)
        nBaseCumSum = [0; cumsum(nBase)];
    else
        nBaseCumSum = [0, cumsum(nBase)];
    end
    groundtruth = cell(instCntQuery, 1);
    for clsInd = 1 : numel(nQuery)
        queryIndEnd = queryIndBeg + nQuery(clsInd) - 1;
        nBaseIndBeg = nBaseCumSum(clsInd + 0) + 1;
        nBaseIndEnd = nBaseCumSum(clsInd + 1) + 0;
        groundtruth(queryIndBeg : queryIndEnd) = {(nBaseIndBeg : nBaseIndEnd)};
        queryIndBeg = queryIndEnd + 1;
    end
elseif ~isempty(linkGt)
    instCntBase = max(linkGt(:));
    instCntQuery = size(linkGt, 1);
    groundtruth = cell(instCntQuery, 1);
    for queryInd = 1 : instCntQuery
        groundtruth{queryInd} = linkGt(queryInd, :);
    end
else
    fprintf('FATAL ERROR: invalid input parameters for evaluation\n');
end

pos = [10, 50 : 50 : instCntBase];
poslen = length(pos);
test_num = instCntQuery;

recall = zeros(length(loopbits), poslen);
precision = zeros(length(loopbits), poslen);
pre_lookup = zeros(length(loopbits), 2);
suc_lookup = zeros(length(loopbits), 2);
ii = 0;
ap = zeros(numel(loopbits), instCntQuery);
for bits = loopbits
    ii = ii + 1;
    fprintf('evaluating results of %d-Bit hash code\n', bits);
    
    % load hash code for query and database
    codeQryPath = sprintf('%s/codeQry.%d.mat', codeDir, bits);
    codeDtbPath = sprintf('%s/codeDtb.%d.mat', codeDir, bits);
    load(codeQryPath);
    load(codeDtbPath);
    D = mex_CalcHammDist(codeQry', codeDtb');
    
    % Evaluation
    label_r = zeros(1, poslen);
    label_p = zeros(1, poslen);
    for n = 1 : instCntQuery
        % compute your distance
        D_code = D(n,:);
        D_truth = groundtruth{n};%ground truth
        for radius = 2 : 3
            ix = find(D_code <= (radius+0.00001));
            % suc_lookup(radius) = suc_lookup(radius) + (~isempty(ix));
            trues = ismember(ix, D_truth);
            ntrues = sum(trues);
            pre_lookup(ii,radius-1) = pre_lookup(ii,radius-1) + ntrues / (length(ix)+1e-5);
        end

        [P, R, apval] = CalcPrecallAP(D_code, D_truth, pos);
        
        ap(ii,n) = apval;
        label_r = label_r + R(1:poslen);
        label_p = label_p + P(1:poslen);
    end

    for radius = 2 : 3
        temp=D<=(radius+0.00001);
        temp=sum(temp,2);
        temp=sum(temp>0);
        suc_lookup(ii,radius-1)=temp/test_num;
    end

    recall(ii,:)=label_r/test_num;
    pre_lookup(ii,:) = pre_lookup(ii,:)/test_num;
    precision(ii,:)=label_p/test_num;
    
    fprintf('MAP = %f\n', mean(ap(ii, :)));
end
MAPval = mean(ap, 2);

evaluation.recall = recall;
evaluation.precision = precision;
evaluation.pre_lookup = pre_lookup;
evaluation.suc_lookup = suc_lookup;
evaluation.MAPval = MAPval;
evaluation.loopbits = loopbits;

save([rsltDir, '/evaluation.mat'], 'evaluation');

toc;

end