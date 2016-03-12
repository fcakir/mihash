function PrepData_Uns()

% initialize constants
scGlobalInit;

% check <kDatasetName>
assert(strcmp(kDatasetName, 'GIST') == 1);

% check <kDataTrnType> (must be 'Stream')
assert(strcmp(kDataTrnType, 'Stream') == 1);

% load original database
instFeat = importdata(kDataDtbPath);
instLabel = importdata(kDataDtbLblPath);

% determine the incoming order of training data (uniform/by-class)
instCntAll = numel(instLabel);
if kInstSmpByCls % sample training data by class
    instIndBeg = 1;
    instIndLst = zeros(instCntAll, 1);
    for clsInd = 1 : kClsCnt
        instIndInCls = find(instLabel == clsInd);
        instCntInCls = numel(instIndInCls);
        instIndEnd = instIndBeg + instCntInCls - 1;
        instIndLst(instIndBeg : instIndEnd) = instIndInCls(randperm(instCntInCls));
        instIndBeg = instIndEnd + 1;
    end
else % sample training data by uniform distribution (across class)
    instIndLst = randperm(instCntAll);
end

% package training data into batches
dbTrnPathLst = cell(kBatchCnt, 1);
for batchInd = 1 : kBatchCnt
    fprintf('batchInd = %d\n', batchInd);
    % determine the start/finish index of current batch
    if batchInd < kBatchCnt % not the last batch
        instIndBeg = (batchInd - 1) * kInstCntInBatch + 1;
        instIndEnd = batchInd * kInstCntInBatch;
    else % collect all remaining instances (maybe less than required)
        instIndBeg = (batchInd - 1) * kInstCntInBatch + 1;
        instIndEnd = instCntAll;
    end

    % save current training batch
    dbTrnInBatch = instFeat(instIndLst(instIndBeg : instIndEnd), :);
    dbTrnPathLst{batchInd} = sprintf('%s/%s.Trn.%03d.mat', kDataDir, kDatasetName, batchInd);
    save(dbTrnPathLst{batchInd}, 'dbTrnInBatch');
end
save(kDataTrnLstPath, 'dbTrnPathLst');

end