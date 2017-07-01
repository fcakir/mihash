% startup script that load the vlfeat library
if isempty(strfind(computer, 'WIN'))
    addpath('/research/humanpose/Libs/vlfeat/toolbox');
else
    addpath('\\kraken\humanpose\Libs\vlfeat\toolbox');
end
vl_setup;
disp('@startup: VLFeat ready');
addpath([pwd '/util']);
