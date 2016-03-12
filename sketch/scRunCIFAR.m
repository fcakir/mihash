close all; clear all; clc;

% write XML file for CIFAR dataset
setupCIFAR.datasetName = 'CIFAR';
setupCIFAR.dataTrnType = 'Solid';
setupCIFAR.clsCnt = 10;
setupCIFAR.batchCnt = 100;
setupCIFAR.instFeatDimCnt = 512;
setupCIFAR.instCntInBatch = 590;
setupCIFAR.evaluationType = 'All';
SaveXMLFile(setupCIFAR, './Config.xml');

% execute each separate functions
PrepData_Sup;
LearnHash_LSH;
LearnHash_OKH;
LearnHash_OSH;
CalcMAPVal;
PlotMAPVal;