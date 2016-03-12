function PrepData_Syn_MVU()

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

% initialize hyber-parameters of MVU distribution
kFeatVecScl = 2.0;

% randomly generate parameters for each MVU distribution
mvuParaLst = cell(kClsCnt, 1);
for clsInd = 1 : kClsCnt
    fprintf('generating parameters for the %d-th MVU distribution\n', clsInd);
    featVecBnd = sort(rand(2, kInstFeatDimCnt) - 0.5, 1) * kFeatVecScl;
    mvuParaLst{clsInd} = struct(...
        'featVecMin', featVecBnd(1, :), ...
        'featVecMax', featVecBnd(2, :));
end

% generate training instances in each batch
dbTrnPathLst = cell(kBatchCnt, 1);
dbTrnInBatch = zeros(kInstCntInBatch, kInstFeatDimCnt, 'single');
for batchInd = 1 : kBatchCnt
    fprintf('generating training instances in the %d-th batch\n', batchInd);
    % generate training batch
    if kInstSmpByCls
        clsInd = ceil(batchInd / kBatchCntInCls);
        dbTrnInBatch = GenRndWrtDstr(mvuParaLst{clsInd}, kInstCntInBatch, 'MVU');
    else
        instRatInCls = rand(kClsCnt, 1);
        instCntInCls = floor(instRatInCls / sum(instRatInCls) * kBatchCntInCls);
        clsIndSel = ceil(rand() * kClsCnt);
        instCntInCls(clsIndSel) = instCntInCls(clsIndSel) + (kBatchCntInCls - sum(instCntInCls));
        instCntInClsCum = [0; cumsum(instCntInCls)];
        for clsInd = 1 : kClsCnt
            instIndBeg = instCntInClsCum(clsInd + 0) + 1;
            instIndEnd = instCntInClsCum(clsInd + 1) + 0;
            dbTrnInBatch(instIndBeg : instIndEnd, :) = GenRndWrtDstr(mvuParaLst{clsInd}, instCntInCls(clsInd), 'MVU');
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
    dbQry(clsInd).feat = GenRndWrtDstr(mvuParaLst{clsInd}, kInstCntQryInCls, 'MVU');
    dbQry(clsInd).label = ones(kInstCntQryInCls, 1) * clsInd;

    % add instances for database
    dbDtb(clsInd).feat = GenRndWrtDstr(mvuParaLst{clsInd}, kInstCntDtbInCls, 'MVU');
    dbDtb(clsInd).label = ones(kInstCntDtbInCls, 1) * clsInd;
end
save(kDataQryPath, 'dbQry');
save(kDataDtbPath, 'dbDtb');

end