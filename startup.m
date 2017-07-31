% startup script that load the vlfeat library
addpath([pwd '/util']);
addpath('./vlfeat/toolbox');
vl_setup;
myLogInfo('VLFeat ready');
