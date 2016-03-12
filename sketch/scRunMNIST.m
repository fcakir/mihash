close all; clear all; clc;

% write XML file for MNIST dataset
setupMNIST.datasetName = 'MNIST';
setupMNIST.dataTrnType = 'Solid';
setupMNIST.clsCnt = 10;
setupMNIST.batchCnt = 100;
setupMNIST.instFeatDimCnt = 784;
setupMNIST.instCntInBatch = 690;
setupMNIST.evaluationType = 'All';
SaveXMLFile(setupMNIST, './Config.xml');

% execute each separate functions
PrepData_Sup;
LearnHash_LSH;
LearnHash_OKH;
LearnHash_OSH;
CalcMAPVal;
PlotMAPVal;