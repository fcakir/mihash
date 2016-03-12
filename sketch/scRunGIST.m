close all; clear all; clc;

% write XML file for GIST dataset
setupGIST.datasetName = 'GIST';
setupGIST.dataTrnType = 'Stream';
setupGIST.clsCnt = 10;
setupGIST.batchCnt = 100;
setupGIST.instFeatDimCnt = 960;
setupGIST.instCntInBatch = 10000;
setupGIST.evaluationType = 'Sel';
SaveXMLFile(setupGIST, './Config.xml');

% execute each separate functions
PrepData_Uns;
LearnHash_LSH;
LearnHash_OKH;
LearnHash_OSH;
CalcMAPVal;
PlotMAPVal;