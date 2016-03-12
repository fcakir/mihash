function PlotMAPVal()

% initialize constants
scGlobalInit;

% parameter for plotting MAP score curves
kMthdClrLst = [{'r'}, {'g'}, {'b'}];
kBitsLinLst = [{'-o'}, {'-+'}, {'-x'}, {'-*'}];

% collect each method's MAP score
mapValLst = importdata(kRsltMAPValPath);

% plot each method's MAP score curve
hold on;
annoStrLst = cell(numel(kLoopBitsLst) * numel(kMthdNameLst), 1);
for curveInd = 1 : numel(kLoopBitsLst) * numel(kMthdNameLst)
    mthdNameInd = uint8(ceil(curveInd / numel(kLoopBitsLst)));
    loopBitsInd = curveInd - numel(kLoopBitsLst) * (mthdNameInd - 1);
    annoStrLst{curveInd} = sprintf('%s - %d Bits', kMthdNameLst{mthdNameInd}, kLoopBitsLst(loopBitsInd));
    
    mthdClr = kMthdClrLst{mthdNameInd};
    bitsLin = kBitsLinLst{loopBitsInd};
    plot(kMAPPosLst, mapValLst(:, loopBitsInd, mthdNameInd)', [mthdClr, bitsLin]);
end
hold off;
grid on;
set(gca, 'XTick', kMAPPosLst);
legend(annoStrLst, 'Location', 'NorthEastOutside');

end