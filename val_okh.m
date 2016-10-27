function val_id = val_okh(c_p, alpha_p, val_dir, val_size, varargin)
% varargin can contain any parameters excluding method specific ones
if isempty(val_dir)
	val_dir = '/research/object_detection/cachedir/online-hashing/okh/val_diaries';
	if ~exist(val_dir, 'dir')
		unix(['mkdir ' val_dir]);
		unix(['chmod g+w ' val_dir]);
		unix(['chmod o-w ' val_dir]);
	end
end
vrg = varargin{1};
dataset = vrg{2};
val_id = sprintf('%s/VAL-OKH-%s-VS%d-C%s-A%s', val_dir, dataset, val_size, ...
	strjoin_fe(strread(num2str(c_p),'%s'),'_'), ...
		strjoin_fe(strread(num2str(alpha_p),'%s'),'_'));
	
fclose('all');
[val_fid, msg] = fopen([val_id '.txt'],'w+');
if val_fid == -1
     error(msg);
end
% get demo_adapthash input
for c=1:length(c_p)
	for a=1:length(alpha_p)
		vrg{end+1} = 'c';
		vrg{end+1} = c_p(c);
		vrg{end+1} = 'alpha';
		vrg{end+1} = alpha_p(a);
		vrg{end+1} = 'val_size';
		vrg{end+1} = val_size;
		varargin_ = vrg;
		[resfn, dp] = demo_okh(varargin_{:});
		r = load(resfn);
		fprintf(val_fid, 'mean mAP: %g c: %d alpha: %d diary path: %s\n', ...
                mean(r.res(:,end)), c_p(c), alpha_p(a), dp);
		clear r
		vrg = varargin{1};
	end
end
fclose(val_fid);
end

