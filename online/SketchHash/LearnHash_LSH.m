function LearnHash_LSH()

% enable diary output
timeCur = datestr(clock, 'dd-mmm-yyyy_HH:MM:SS');
kLogFilePath = sprintf('./Log.Files/LSH.%s.log', timeCur);
diary(kLogFilePath);
diary on;

% initialize constants
scGlobalInit;

% remove preivous codes/results
DeltCodeRslt(kCodeDir_LSH, kRsltDir_LSH);

% load pre-splitted database
[instFeatQry, instFeatDtb, instCntInClsQry, instCntInClsDtb, instLinkQryGt] = LoadInstQryDtb();

% load training batches (or their indexes)
if strcmp(kDataTrnType, 'Solid')
    dbTrn = importdata(kDataTrnPath);
    batchCnt = numel(dbTrn);
elseif strcmp(kDataTrnType, 'Stream')
    dbTrnPathLst = importdata(kDataTrnLstPath);
    batchCnt = numel(dbTrnPathLst);
end

% compute the mean feature vector
instCntSeen = 0;
instFeatAve = zeros(1, kInstFeatDimCnt);
for batchInd = 1 : batchCnt
    fprintf('updating mean feature vector with the %d-th batch\n', batchInd);
    
    % obtain training data in the batch
    if strcmp(kDataTrnType, 'Solid')
        instFeatInBatch = dbTrn{batchInd};
    elseif strcmp(kDataTrnType, 'Stream')
        instFeatInBatch = importdata(dbTrnPathLst{batchInd});
    end
    instCntInBatch = size(instFeatInBatch, 1);
    
    % update mean feature vector
    instFeatAve = (instFeatAve * instCntSeen + sum(instFeatInBatch, 1)) / (instCntSeen + instCntInBatch);
    instCntSeen = instCntSeen + instCntInBatch;
end

% compute the zero-mean query/database instances
instFeatQryCen = bsxfun(@minus, instFeatQry, instFeatAve);
instFeatDtbCen = bsxfun(@minus, instFeatDtb, instFeatAve);

% run LSH method
hashProjMat = cell(numel(kLoopBitsLst));
for ind = 1 : numel(kLoopBitsLst)
    % determine hashing functions
    bits = kLoopBitsLst(ind);
    hashProjMat{ind} = normrnd(0, 1, kInstFeatDimCnt, bits);
    
    % compute hash code for query/database subset
    codeQry = (instFeatQryCen * hashProjMat{ind} > 0);
    codeDtb = (instFeatDtbCen * hashProjMat{ind} > 0);

    % save hash code for query/database subset
    codeQryPath = sprintf('%s/codeQry.%d.mat', kCodeDir_LSH, bits);
    codeDtbPath = sprintf('%s/codeDtb.%d.mat', kCodeDir_LSH, bits);
    save(codeQryPath, 'codeQry');
    save(codeDtbPath, 'codeDtb');
end

% evaluate the final hash code
rsltLstFnl = Evaluate_Fast(instCntInClsDtb, instCntInClsQry, instLinkQryGt, kLoopBitsLst, kCodeDir_LSH, kRsltDir_LSH);
fprintf('MAP = %f\n', rsltLstFnl.MAPval);

% disable diary output
diary off;

end