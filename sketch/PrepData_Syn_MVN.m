function PrepData_Syn_MVN()

% initialize constants
scGlobalInit;

% check <kDatasetName>
assert(strcmp(kDatasetName, 'SYNTC') == 1);

% check <kDataTrnType> (must be 'Stream')
assert(strcmp(kDataTrnType, 'Stream') == 1);

% remove previously generated data
delete([kDataDir, '/*.mat']);

% set the size of batch/query/database
kBatchCntInCls = kBatchCnt / kClsCnt; % number of training batches per class
kInstCntQryInCls = 20; % number of query instances per class
kInstCntDtbInCls = 200; % number of database instances per class

% initialize hyber-parameters of MVN distribution
kMeanVecScl = 1.0;
kCovrMatScl = 0.3;

% randomly generate parameters for each MVN distribution
mvnParaLst = cell(kClsCnt, 1);
for clsInd = 1 : kClsCnt
    fprintf('generating parameters for the %d-th MVN distribution\n', clsInd);
    meanVec = (rand(1, kInstFeatDimCnt) - 0.5) * kMeanVecScl;
    covrMat = rand(kInstFeatDimCnt) * kCovrMatScl;
    mvnParaLst{clsInd} = struct(...
        'meanVec', single(meanVec), ...
        'covrMat', single(covrMat' * covrMat));
end

% generate training instances in each batch
dbTrnPathLst = cell(kBatchCnt, 1);
dbTrnInBatch = zeros(kInstCntInBatch, kInstFeatDimCnt, 'single');
for batchInd = 1 : kBatchCnt
    fprintf('generating training instances in the %d-th batch\n', batchInd);
    % generate training batch
    if kInstSmpByCls
        clsInd = ceil(batchInd / kBatchCntInCls);
        dbTrnInBatch = GenRndWrtDstr(mvnParaLst{clsInd}, kInstCntInBatch, 'MVN');
    else
        instRatInCls = rand(kClsCnt, 1);
        instCntInCls = floor(instRatInCls / sum(instRatInCls) * kBatchCntInCls);
        clsIndSel = ceil(rand() * kClsCnt);
        instCntInCls(clsIndSel) = instCntInCls(clsIndSel) + (kBatchCntInCls - sum(instCntInCls));
        instCntInClsCum = [0; cumsum(instCntInCls)];
        for clsInd = 1 : kClsCnt
            instIndBeg = instCntInClsCum(clsInd + 0) + 1;
            instIndEnd = instCntInClsCum(clsInd + 1) + 0;
            dbTrnInBatch(instIndBeg : instIndEnd, :) = GenRndWrtDstr(mvnParaLst{clsInd}, instCntInCls(clsInd), 'MVN');
        end
    end

    % save current training batch
    dbTrnInBatch = single(dbTrnInBatch);
    dbTrnPathLst{batchInd} = sprintf('%s/%s.Trn.%03d.mat', kDataDir, kDatasetName, batchInd);
    save(dbTrnPathLst{batchInd}, 'dbTrnInBatch');
end
save(kDataTrnLstPath, 'dbTrnPathLst');

% generate query/database instances for each class
dbQry(1 : kClsCnt) = struct(...
    'feat', zeros(kInstCntQryInCls, kInstFeatDimCnt, 'single'), ...
    'label', zeros(kInstCntQryInCls, 1));
dbDtb(1 : kClsCnt) = struct(...
    'feat', zeros(kInstCntDtbInCls, kInstFeatDimCnt, 'single'), ...
    'label', zeros(kInstCntDtbInCls, 1));
for clsInd = 1 : kClsCnt
    fprintf('generating query/database instances for %d-th class\n', clsInd);

    % generate instances to query subset
    dbQry(clsInd).feat = GenRndWrtDstr(mvnParaLst{clsInd}, kInstCntQryInCls, 'MVN');
    dbQry(clsInd).label = ones(kInstCntQryInCls, 1) * clsInd;

    % add instances for database
    dbDtb(clsInd).feat = GenRndWrtDstr(mvnParaLst{clsInd}, kInstCntDtbInCls, 'MVN');
    dbDtb(clsInd).label = ones(kInstCntDtbInCls, 1) * clsInd;
end
save(kDataQryPath, 'dbQry');
save(kDataDtbPath, 'dbDtb');

end