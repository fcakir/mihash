function distMat = CalcDistMat(instFeat_1, instFeat_2)

instCnt_1 = size(instFeat_1, 1);
instCnt_2 = size(instFeat_2, 1);
distMat = repmat(sum(instFeat_1 .^ 2, 2), [1, instCnt_2]) + repmat(sum(instFeat_2 .^ 2, 2)', [instCnt_1, 1]) - 2 * instFeat_1 * instFeat_2';

end