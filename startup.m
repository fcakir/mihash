addpath('/research/humanpose/Libs/vlfeat/toolbox');
vl_setup;
disp('@startup: VLFeat ready');
%matlabpool 6;  % now handle it in get_opts
