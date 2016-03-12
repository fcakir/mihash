close all; clear all; clc;

% write XML file for SYNTC dataset
setupSYNTC.datasetName = 'SYNTC';
setupSYNTC.dataTrnType = 'Stream';
setupSYNTC.clsCnt = 100;
setupSYNTC.batchCnt = 100;
setupSYNTC.instFeatDimCnt = 10000;
setupSYNTC.instCntInBatch = 1000;
setupSYNTC.evaluationType = 'Sel';
SaveXMLFile(setupSYNTC, './Config.xml');

% execute each separate functions
PrepData_Syn_MVU;
LearnHash_LSH;
LearnHash_OKH;
LearnHash_OSH;
CalcMAPVal;
PlotMAPVal;