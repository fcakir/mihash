% this file contains shared constant variables for hashing function learning

% basic setting across different datasets
kInstSmpByCls = true; % sample training data class-by-class/across-class
kLoopBitsLst = [16, 32, 64]; % list of possible hashing code length
kMthdNameLst = [{'LSH'}, {'OKH'}, {'OSH'}]; % name of methods to be evaluated

% load dataset-specific setup
setupStr = LoadXMLFile('./Config.xml');
kDatasetName = setupStr.datasetName; % name of dataset bein used
kDataTrnType = setupStr.dataTrnType; % 'Solid' - all packed in one file; 'Stream' - one file for one batch
kClsCnt = setupStr.clsCnt; % number of classes/clusters
kBatchCnt = setupStr.batchCnt; % number of training batches
kInstFeatDimCnt = setupStr.instFeatDimCnt; % number of feature dimensions
kInstCntInBatch = setupStr.instCntInBatch; % number of instances in each incoming batch
kMAPPosLst = setupStr.mapPosLst; % evaluation positions

% set-up directories for dataset I/O
kDataDir = sprintf('./Data/%s', kDatasetName);
kDataRawPath = sprintf('%s/%s.Raw.mat', kDataDir, kDatasetName);
kDataTrnPath = sprintf('%s/%s.Trn.mat', kDataDir, kDatasetName);
kDataTrnLstPath = sprintf('%s/%s.Trn.Lst.mat', kDataDir, kDatasetName);
kDataQryPath = sprintf('%s/%s.Qry.mat', kDataDir, kDatasetName);
kDataDtbPath = sprintf('%s/%s.Dtb.mat', kDataDir, kDatasetName);
kDataQryGtPath = sprintf('%s/%s.Qry.Gt.mat', kDataDir, kDatasetName);
kDataDtbLblPath = sprintf('%s/%s.Dtb.Lbl.mat', kDataDir, kDatasetName);

% set-up directories for binary code I/O
kCodeMainDir = sprintf('./Code/%s', kDatasetName);
kCodeDir_LSH = sprintf('%s/LSH', kCodeMainDir);
kCodeDir_OKH = sprintf('%s/OKH', kCodeMainDir);
kCodeDir_OSH = sprintf('%s/OSH', kCodeMainDir);

% set-up directories for evaluation result I/O
kRsltMainDir = sprintf('./Rslt/%s', kDatasetName);
kRsltMAPValPath = sprintf('%s/mapValLst.mat', kRsltMainDir);
kRsltDir_LSH = sprintf('%s/LSH', kRsltMainDir);
kRsltDir_OKH = sprintf('%s/OKH', kRsltMainDir);
kRsltDir_OSH = sprintf('%s/OSH', kRsltMainDir);
kRsltTimePath_OKH = sprintf('%s/timeElpsStr.mat', kRsltDir_OKH);
kRsltTimePath_OSH = sprintf('%s/timeElpsStr.mat', kRsltDir_OSH);