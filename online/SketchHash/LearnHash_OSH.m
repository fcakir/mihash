function LearnHash_OSH()

% enable diary output
timeCur = datestr(clock, 'dd-mmm-yyyy_HH:MM:SS');
kLogFilePath = sprintf('./Log.Files/OSH.%s.log', timeCur);
diary(kLogFilePath);
diary on;

% initialize constants
scGlobalInit;

% remove preivous codes/results
DeltCodeRslt(kCodeDir_OSH, kRsltDir_OSH);

% initialize parameters for OSH learning
paraOSH.instCntSkc = 100;

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

% initialize hashing functions
hashProjMat = cell(numel(kLoopBitsLst), 1);
for ind = 1 : numel(kLoopBitsLst)
    bits = kLoopBitsLst(ind);
    hashProjMat{ind} = rand(kInstFeatDimCnt, bits) - 0.5;
end

% create a variable to record time consuming
timeElpsStr = struct(...
    'loadBtchData', zeros(batchCnt, 1), ...
    'updtHashFunc', zeros(batchCnt, 1), ...
    'calcHashCode', zeros(batchCnt, 1));

% run online hashing methods
instCntSeen = 0;
instFeatAvePre = zeros(1, kInstFeatDimCnt);
instFeatSkc = [];
for batchInd = 1 : batchCnt
    fprintf('update hashing function with the %d-th batch\n', batchInd);
    
    %%%%%%%%%% LOAD BATCH DATA - BELOW %%%%%%%%%%
    tic;
    
    % create directory to store hash code and evaluation result
    codeDirCur = sprintf('%s/%03d', kCodeDir_OSH, batchInd);
    rsltDirCur = sprintf('%s/%03d', kRsltDir_OSH, batchInd);
    mkdir(codeDirCur);
    mkdir(rsltDirCur);
    
    % obtain training data in the batch
    if strcmp(kDataTrnType, 'Solid')
        instFeatInBatch = dbTrn{batchInd};
    elseif strcmp(kDataTrnType, 'Stream')
        instFeatInBatch = importdata(dbTrnPathLst{batchInd});
    end
    instCntInBatch = size(instFeatInBatch, 1);
    
    timeElpsStr.loadBtchData(batchInd) = toc;
    %%%%%%%%%% LOAD BATCH DATA - ABOVE %%%%%%%%%%
    
    %%%%%%%%%% UPDATE HASHING FUNCTION - BELOW %%%%%%%%%%
    tic;
    
    % calculate current mean feature vector
    instFeatAveCur = mean(instFeatInBatch, 1);
    
    % sketech current training batch
    if batchInd == 1
        instFeatToSkc = bsxfun(@minus, instFeatInBatch, instFeatAveCur);
    else
        instFeatCmps = sqrt(instCntSeen * instCntInBatch / (instCntSeen + instCntInBatch)) * (instFeatAveCur - instFeatAvePre);
        instFeatToSkc = [bsxfun(@minus, instFeatInBatch, instFeatAveCur); instFeatCmps];
    end
    instFeatSkc = MatrixSketch_Incr(instFeatSkc, instFeatToSkc, paraOSH.instCntSkc);
    
    % update mean feature vector and instance counter
    instFeatAvePre = (instFeatAvePre * instCntSeen + instFeatAveCur * instCntInBatch) / (instCntSeen + instCntInBatch);
    instCntSeen = instCntSeen + instCntInBatch;
    
    % compute QR decomposition of the sketched matrix
    [q, r] = qr(instFeatSkc', 0);
    [u, ~, ~] = svd(r, 'econ');
    v = q * u;
    
    for ind = 1 : numel(kLoopBitsLst)
        % obtain the length of hashing code
        bits = kLoopBitsLst(ind);
        
        % obtain the original projection matrix
        hashProjMatOrg = v(:, 1 : bits);
        
        % use random rotation
        R = orth(randn(bits));

        % update hashing function
        hashProjMat{ind} = single(hashProjMatOrg * R);
    end
    
    timeElpsStr.updtHashFunc(batchInd) = toc;
    %%%%%%%%%% UPDATE HASHING FUNCTION - ABOVE %%%%%%%%%%
    
    %%%%%%%%%% COMPUTE HASHING CODE - BELOW %%%%%%%%%%
    tic;
    
    % compute centered query/database instances
    instFeatQryCen = bsxfun(@minus, instFeatQry, instFeatAvePre);
    instFeatDtbCen = bsxfun(@minus, instFeatDtb, instFeatAvePre);
    
    % compute hash code for query/database subset
    for ind = 1 : numel(kLoopBitsLst)
        bits = kLoopBitsLst(ind);
        codeQry = (instFeatQryCen * hashProjMat{ind} > 0);
        codeDtb = (instFeatDtbCen * hashProjMat{ind} > 0);

        % save hash code for query/database subset
        codeQryPath = sprintf('%s/codeQry.%d.mat', codeDirCur, bits);
        codeDtbPath = sprintf('%s/codeDtb.%d.mat', codeDirCur, bits);
        save(codeQryPath, 'codeQry');
        save(codeDtbPath, 'codeDtb');
    end
    
    timeElpsStr.calcHashCode(batchInd) = toc;
    %%%%%%%%%% COMPUTE HASHING CODE - ABOVE %%%%%%%%%%
end
save(kRsltTimePath_OSH, 'timeElpsStr');
fprintf('average time for loading batch data: %f\n', mean(timeElpsStr.loadBtchData));
fprintf('average time for updating hashing function: %f\n', mean(timeElpsStr.updtHashFunc));
fprintf('average time for computing hashing code: %f\n', mean(timeElpsStr.calcHashCode));

% evaluate the final hash code
codeDirFnl = sprintf('%s/%03d', kCodeDir_OSH, batchCnt);
rsltDirFnl = sprintf('%s/%03d', kRsltDir_OSH, batchCnt);
rsltStrFnl = Evaluate_Fast(instCntInClsDtb, instCntInClsQry, instLinkQryGt, kLoopBitsLst, codeDirFnl, rsltDirFnl);
fprintf('MAP = %f\n', rsltStrFnl.MAPval);

% disable diary output
diary off;

end