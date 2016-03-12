function [instFeatQry, instFeatDtb, instCntInClsQry, instCntInClsDtb, instLinkQryGt] = LoadInstQryDtb()

% initialize constants
scGlobalInit;

% load query/database instances
if strcmp(kDatasetName, 'GIST')
    instFeatQry = importdata(kDataQryPath);
    instFeatDtb = importdata(kDataDtbPath);
    instCntInClsQry = [];
    instCntInClsDtb = [];
    instLinkQryGt = importdata(kDataQryGtPath);
else
    dbQry = importdata(kDataQryPath);
    dbDtb = importdata(kDataDtbPath);
    instFeatQry = ExtrInstFeat(dbQry);
    instFeatDtb = ExtrInstFeat(dbDtb);
    instCntInClsQry = arrayfun(@(x)(numel(x.label)), dbQry);
    instCntInClsDtb = arrayfun(@(x)(numel(x.label)), dbDtb);
    instLinkQryGt = [];
end

end