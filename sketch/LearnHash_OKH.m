function LearnHash_OKH()

% enable diary output
timeCur = datestr(clock, 'dd-mmm-yyyy_HH:MM:SS');
kLogFilePath = sprintf('./Log.Files/OKH.%s.log', timeCur);
diary(kLogFilePath);
diary on;

% initialize constants
scGlobalInit;

% remove preivous codes/results
DeltCodeRslt(kCodeDir_OKH, kRsltDir_OKH);

% initialize parameters for OKH learning
paraOKH.c = 0.1;
paraOKH.alpha = 0.2;
paraOKH.anchCnt = 300;
paraOKH.instDistThrsRat = 0.05;
paraOKH.instPoolSiz = 10000;

% load pre-splitted database
[instFeatQry, instFeatDtb, instCntInClsQry, instCntInClsDtb, instLinkQryGt] = LoadInstQryDtb();
instCntQry = size(instFeatQry, 1);
instCntDtb = size(instFeatDtb, 1);

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
    hashProjMat{ind} = rand(paraOKH.anchCnt + 1, bits) - 0.5;
end

% create a variable to record time consuming
timeElpsStr = struct(...
    'loadBtchData', zeros(batchCnt, 1), ...
    'updtHashFunc', zeros(batchCnt, 1), ...
    'calcHashCode', zeros(batchCnt, 1));

% run online hashing methods
instCntSeen = 0;
instCntInPool = 0;
instFeatInPool = zeros(paraOKH.instPoolSiz, kInstFeatDimCnt);
for batchInd = 1 : batchCnt
    fprintf('update hashing function with the %d-th batch\n', batchInd);
    
    %%%%%%%%%% LOAD BATCH DATA - BELOW %%%%%%%%%%
    tic;
    
    % create directory to store hash code and evaluation result
    codeDirCur = sprintf('%s/%03d', kCodeDir_OKH, batchInd);
    rsltDirCur = sprintf('%s/%03d', kRsltDir_OKH, batchInd);
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
    
    % update instances in pool (for distance threshold computing)
    instCntSeen = instCntSeen + instCntInBatch;
    if instCntInPool + instCntInBatch <= paraOKH.instPoolSiz
        instFeatInPool(instCntInPool + 1 : instCntInPool + instCntInBatch, :) = instFeatInBatch;
        instCntInPool = instCntInPool + instCntInBatch;
    else
        if instCntInPool < paraOKH.instPoolSiz
            instCntInBatchAppd = paraOKH.instPoolSiz - instCntInPool;
            instFeatInPool(instCntInPool + 1 : end, :) = instFeatInBatch(1 : instCntInBatchAppd, :);
            instCntInBatchRest = instCntInBatch - instCntInBatchAppd;
        else
            instCntInBatchAppd = 0;
            instCntInBatchRest = instCntInBatch;
        end
        
        instAddProb = instCntInBatchRest / instCntSeen;
        instIndLstAdd = find(rand(instCntInBatchRest, 1) < instAddProb) + instCntInBatchAppd;
        instIndLstDlt = randperm(paraOKH.instPoolSiz, numel(instIndLstAdd));
        instFeatInPool(instIndLstDlt, :) = instFeatInBatch(instIndLstAdd, :);
        instCntInPool = paraOKH.instPoolSiz;
    end

    % select instances as anchors
    if batchInd == 1
        % randomly select instances as anchors
        paraOKH.anchLst = instFeatInBatch(randperm(instCntInBatch, paraOKH.anchCnt), :);

        % compute kernel matrix for database subset
        instDistMatToAnch = CalcDistMat(paraOKH.anchLst, instFeatDtb);
        instDistAveToAnch = 2 * mean(instDistMatToAnch(:));
        instKernMatDtb = [exp(-instDistMatToAnch / instDistAveToAnch); ones(1, instCntDtb)];
        
        % compute kernel matrix for query subset
        instDistMatToAnch = CalcDistMat(paraOKH.anchLst, instFeatQry);
        instKernMatQry = [exp(-instDistMatToAnch / instDistAveToAnch); ones(1, instCntQry)];
    end

    % compute instances' distance to seen training instances
    instDistMat = CalcDistMat(instFeatInBatch, instFeatInPool(1 : instCntInPool, :));
    instDistMatSort = sort(instDistMat, 2);
    instDistThrs = instDistMatSort(:, round(instCntInPool * paraOKH.instDistThrsRat));

    % compute instances' distance to anchors
    instDistMatToAnch = CalcDistMat(paraOKH.anchLst, instFeatInBatch);
    instKernMatBth = [exp(-instDistMatToAnch / instDistAveToAnch); ones(1, instCntInBatch)];

    % use each instance pair to update hashing functions
    instPairCntInBatch = floor(instCntInBatch / 2);
    for instPairIndInBatch = 1 : instPairCntInBatch
        instIndInBatch_1 = instPairIndInBatch * 2 - 1;
        instIndInBatch_2 = instPairIndInBatch * 2 - 0;
        instFeatInBatch_1 = instFeatInBatch(instIndInBatch_1, :);
        instFeatInBatch_2 = instFeatInBatch(instIndInBatch_2, :);
        instPairDist = sum((instFeatInBatch_1 - instFeatInBatch_2) .^ 2);
        instPairLabel = (instPairDist <= instDistThrs(instIndInBatch_1));
        instKernInBatch_1 = instKernMatBth(:, instIndInBatch_1);
        instKernInBatch_2 = instKernMatBth(:, instIndInBatch_2);

        % update hashing function
        for ind = 1 : numel(kLoopBitsLst)
            hashProjMat{ind} = OKHLearn(instKernInBatch_1, instKernInBatch_2, instPairLabel, hashProjMat{ind}, paraOKH);
        end
    end
    
    timeElpsStr.updtHashFunc(batchInd) = toc;
    %%%%%%%%%% UPDATE HASHING FUNCTION - ABOVE %%%%%%%%%%
    
    %%%%%%%%%% COMPUTE HASHING CODE - BELOW %%%%%%%%%%
    tic;
    
    % compute hash code for query/database subset
    for ind = 1 : numel(kLoopBitsLst)
        bits = kLoopBitsLst(ind);
        codeQry = (instKernMatQry' * hashProjMat{ind} > 0);
        codeDtb = (instKernMatDtb' * hashProjMat{ind} > 0);

        % save hash code for query/database subset
        codeQryPath = sprintf('%s/codeQry.%d.mat', codeDirCur, bits);
        codeDtbPath = sprintf('%s/codeDtb.%d.mat', codeDirCur, bits);
        save(codeQryPath, 'codeQry');
        save(codeDtbPath, 'codeDtb');
    end
    
    timeElpsStr.calcHashCode(batchInd) = toc;
    %%%%%%%%%% COMPUTE HASHING CODE - ABOVE %%%%%%%%%%
end
save(kRsltTimePath_OKH, 'timeElpsStr');
fprintf('average time for loading batch data: %f\n', mean(timeElpsStr.loadBtchData));
fprintf('average time for updating hashing function: %f\n', mean(timeElpsStr.updtHashFunc));
fprintf('average time for computing hashing code: %f\n', mean(timeElpsStr.calcHashCode));

% evaluate the final hash code
codeDirFnl = sprintf('%s/%03d', kCodeDir_OKH, batchCnt);
rsltDirFnl = sprintf('%s/%03d', kRsltDir_OKH, batchCnt);
rsltStrFnl = Evaluate_Fast(instCntInClsDtb, instCntInClsQry, instLinkQryGt, kLoopBitsLst, codeDirFnl, rsltDirFnl);
fprintf('MAP = %f\n', rsltStrFnl.MAPval);

% disable diary output
diary off;

end