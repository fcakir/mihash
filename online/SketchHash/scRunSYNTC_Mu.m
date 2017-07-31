close all; clear all; clc;

% candidates of hyber parameters
kInstFeatDimCntLst = [1, 2, 4, 6, 8, 10] * (10 ^ 2);
kInstCntInBatchLst = [1, 2, 4, 6, 8, 10] * (10 ^ 4);

% run OSH under each setting
for instFeatDimCnt = kInstFeatDimCntLst
    for instCntInBatch = kInstCntInBatchLst
        % write XML file for SYNTC dataset
        setupSYNTC.datasetName = 'SYNTC';
        setupSYNTC.dataTrnType = 'Stream';
        setupSYNTC.clsCnt = 100;
        setupSYNTC.batchCnt = 100;
        setupSYNTC.instFeatDimCnt = instFeatDimCnt;
        setupSYNTC.instCntInBatch = instCntInBatch;
        setupSYNTC.evaluationType = 'Sel';
        SaveXMLFile(setupSYNTC, './Config.xml');
        
        % execute each separate functions
        PrepData_Syn_MVU;
        LearnHash_OSH;
    end
end