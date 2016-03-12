close all; clear all; clc;

% initialize constants
GlobalInit;

% check <kDataTrnType> (must be 'Stream')
assert(strcmp(kDataTrnType, 'Stream') == 1);

% load original database
instFeat = importdata(kDataDtbPath);

% run K-means clustering to divide the database
optionsKmeans = statset('MaxIter', 10);
instLabel = kmeans(instFeat, kClsCnt, 'Options', optionsKmeans);
save(kDataDtbLblPath, 'instLabel');