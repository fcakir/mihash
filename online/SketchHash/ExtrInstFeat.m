function instFeatLst = ExtrInstFeat(dbCIFAR)

isFirstEntry = true;
instCnt = sum(arrayfun(@(x)(numel(x.label)), dbCIFAR));
for clsInd = 1 : numel(dbCIFAR)
    if ~isempty(dbCIFAR(clsInd).feat)
        % allocate memory if it is the first class with instances existing
        if isFirstEntry
            isFirstEntry = false;
            instFeatDimCnt = size(dbCIFAR(clsInd).feat, 2);
            instFeatLst = zeros(instCnt, instFeatDimCnt);
            instIndBeg = 1;
        end
        % copy existing instances to <instFeatLst>
        instCntInCls = numel(dbCIFAR(clsInd).label);
        instIndEnd = instIndBeg + instCntInCls - 1;
        instFeatLst(instIndBeg : instIndEnd, :) = dbCIFAR(clsInd).feat;
        instIndBeg = instIndEnd + 1;
    end
end

end