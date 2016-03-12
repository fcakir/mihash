function CalcMAPVal()

% initialize constants
scGlobalInit;

% load pre-splitted database
[~, ~, instCntInClsQry, instCntInClsDtb, instLinkQryGt] = LoadInstQryDtb();

% collect each method's MAP score
mapValLst = zeros(numel(kMAPPosLst), numel(kLoopBitsLst), numel(kMthdNameLst));
for ind = 1 : numel(kMAPPosLst)
    for mthdNameInd = 1 : numel(kMthdNameLst)
        batchInd = kMAPPosLst(ind);
        mthdName = kMthdNameLst{mthdNameInd};
        fprintf('evaluating %s results after %d batches\n', mthdName, batchInd);
        
        if strcmp(mthdName, 'LSH')
            if batchInd == 1
                codeDir = sprintf('%s/%s', kCodeMainDir, mthdName);
                rsltDir = sprintf('%s/%s', kRsltMainDir, mthdName);
                rsltStr = Evaluate_Fast(instCntInClsDtb, instCntInClsQry, instLinkQryGt, kLoopBitsLst, codeDir, rsltDir);
                mapValLst(:, :, mthdNameInd) = repmat(rsltStr.MAPval', [numel(kMAPPosLst), 1]);
            else % only one evaluation needed
                continue;
            end
        else
            codeDir = sprintf('%s/%s/%03d', kCodeMainDir, mthdName, batchInd);
            rsltDir = sprintf('%s/%s/%03d', kRsltMainDir, mthdName, batchInd);
            rsltStr = Evaluate_Fast(instCntInClsDtb, instCntInClsQry, instLinkQryGt, kLoopBitsLst, codeDir, rsltDir);
            mapValLst(ind, :, mthdNameInd) = rsltStr.MAPval';
        end
    end
end
save(kRsltMAPValPath, 'mapValLst');

end