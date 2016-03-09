if isempty(strfind(computer, 'WIN'))
	addpath('/research/humanpose/Libs/vlfeat/toolbox');
else
	addpath('\\kraken\humanpose\Libs\vlfeat\toolbox');
end
vl_setup;
disp('@startup: VLFeat ready');
%matlabpool 6;  % now handle it in get_opts
