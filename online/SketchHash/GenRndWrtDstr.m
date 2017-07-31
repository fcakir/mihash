function instFeat = GenRndWrtDstr(rndParaStr, instCnt, rndStr)

if strcmp(rndStr, 'MVN')
    instFeat = mvnrnd(rndParaStr.meanVec, rndParaStr.covrMat, instCnt);
elseif strcmp(rndStr, 'MVU')
    instFeatDimCnt = numel(rndParaStr.featVecMin);
    instFeat_1 = rand(instCnt, instFeatDimCnt);
    instFeat_2 = bsxfun(@times, instFeat_1, rndParaStr.featVecMax - rndParaStr.featVecMin);
    instFeat = bsxfun(@plus, instFeat_2, rndParaStr.featVecMin);
else
    fprintf('unrecognized random distribution: %s\n', rndStr);
end

% convert to single to save storage
instFeat = single(instFeat);

end