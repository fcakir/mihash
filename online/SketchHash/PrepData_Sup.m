function PrepData_Sup()

% initialize constants
scGlobalInit;

% check <kDatasetName>
assert((strcmp(kDatasetName, 'CIFAR') == 1) || (strcmp(kDatasetName, 'MNIST') == 1));

% load original database
dbRaw = importdata(kDataRawPath);

% re-map labels in <dbRaw.train_label> and <dbRaw.test_label> to [1, C]
clsIndLstOld = unique(dbRaw.train_label);
[~, clsIndLstNew] = sort(clsIndLstOld);
dbRaw.train_label = changem(dbRaw.train_label, clsIndLstNew, clsIndLstOld);
dbRaw.test_label = changem(dbRaw.test_label, clsIndLstNew, clsIndLstOld);

% determine the incoming order of training data (uniform/by-class)
instCntAll = numel(dbRaw.train_label);
if kInstSmpByCls % sample training data by class
    instIndBeg = 1;
    instIndLst = zeros(instCntAll, 1);
    for clsInd = 1 : kClsCnt
        instIndInCls = find(dbRaw.train_label == clsInd);
        instCntInCls = numel(instIndInCls);
        instIndEnd = instIndBeg + instCntInCls - 1;
        instIndLst(instIndBeg : instIndEnd) = instIndInCls(randperm(instCntInCls));
        instIndBeg = instIndEnd + 1;
    end
else % sample training data by uniform distribution (across class)
    instIndLst = randperm(instCntAll);
end

% package training data into batches
dbTrn = cell(kBatchCnt, 1);
dbTrnPathLst = cell(kBatchCnt, 1);
for batchInd = 1 : kBatchCnt
    % determine the start/finish index of current batch
    if batchInd < kBatchCnt % not the last batch
        instIndBeg = (batchInd - 1) * kInstCntInBatch + 1;
        instIndEnd = batchInd * kInstCntInBatch;
    else % collect all remaining instances (maybe less than required)
        instIndBeg = (batchInd - 1) * kInstCntInBatch + 1;
        instIndEnd = instCntAll;
    end

    % save current training batch
    if strcmp(kDataTrnType, 'Solid')
        dbTrn{batchInd} = dbRaw.train_x(instIndLst(instIndBeg : instIndEnd), :);
    elseif strcmp(kDataTrnType, 'Stream')
        dbTrnInBatch = dbRaw.train_x(instIndLst(instIndBeg : instIndEnd), :);
        dbTrnPathLst{batchInd} = sprintf('%s/%s.Trn.%03d.mat', kDataDir, kDatasetName, batchInd);
        save(dbTrnPathLst{batchInd}, 'dbTrnInBatch');
    end
end
if strcmp(kDataTrnType, 'Solid')
    save(kDataTrnPath, 'dbTrn');
elseif strcmp(kDataTrnType, 'Stream')
    save(kDataTrnLstPath, 'dbTrnPathLst');
end

% extract instances for training/testing
dbQry(1 : kClsCnt) = struct('feat', [], 'label', []);
dbDtb(1 : kClsCnt) = struct('feat', [], 'label', []);
for clsInd = 1 : kClsCnt
    % add instances to query subset
    instIndQry = find(dbRaw.test_label == clsInd);
    dbQry(clsInd).feat = dbRaw.test_x(instIndQry, :);
    dbQry(clsInd).label = dbRaw.test_label(instIndQry);
    % add instances for database
    instIndDtb = find(dbRaw.train_label == clsInd);
    dbDtb(clsInd).feat = dbRaw.train_x(instIndDtb, :);
    dbDtb(clsInd).label = dbRaw.train_label(instIndDtb);
end
save(kDataQryPath, 'dbQry');
save(kDataDtbPath, 'dbDtb');

end
