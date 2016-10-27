function val_id = val_osh(stepsize_p, val_dir, val_size, varargin)
% varargin can contain any parameters excluding method specific ones
if isempty(val_dir)
	val_dir = '/research/object_detection/cachedir/online-hashing/val_diaries';
	if ~exist(val_dir, 'dir')
		unix(['mkdir ' val_dir]);
		unix(['chmod g+w ' val_dir]);
		unix(['chmod o-w ' val_dir]);
	end
end
vrg = varargin{1};
dataset = vrg{2};
val_id = sprintf('%s/VAL-OSH-%s-VS%d-SS%s', val_dir, dataset, val_size, ...
	strjoin_fe(strread(num2str(stepsize_p),'%s'),'_'));
fclose('all');
[val_fid, msg] = fopen([val_id '.txt'],'w+');
if val_fid == -1
     error(msg);
end
% get demo_osh input
for s=1:length(stepsize_p)
	vrg{end+1} = 'stepsize';
	vrg{end+1} = stepsize_p(s);
	vrg{end+1} = 'val_size';
	vrg{end+1} = val_size;
	varargin_ = vrg;
	[resfn, dp] = demo_osh(varargin_{:});
	r = load(resfn);
	fprintf(val_fid, 'mean mAP: %g stepsize: %d diary path: %s\n', ...
            mean(r.res(:,end)), stepsize_p(s), dp);
	clear r
	vrg = varargin{1};
end
fclose(val_fid);
end
