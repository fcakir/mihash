function val_id =  val_sketch(sketchSize_p, batchSize_p, val_dir, val_size, varargin)
% varargin can contain any parameters excluding method specific ones
if isempty(val_dir)
	val_dir = '/research/object_detection/cachedir/online-hashing/sketch/val_diaries';
	if ~exist(val_dir, 'dir')
		unix(['mkdir ' val_dir]);
		unix(['chmod g+w ' val_dir]);
		unix(['chmod o-w ' val_dir]);
	end
end
vrg = varargin{1};
dataset = vrg{2};
val_id = sprintf('%s/VAL-Sketch-%s-VS%d-SketchSize%s-BatchSize%s', val_dir, dataset, val_size, ...
	strjoin_fe(strread(num2str(sketchSize_p),'%s'),'_'), ...
		strjoin_fe(strread(num2str(batchSize_p),'%s'),'_'));
fclose('all');
[val_fid, msg] = fopen([val_id '.txt'],'w+');
if val_fid == -1
     error(msg);
end
% get demo_adapthash input
for s=1:length(sketchSize_p)
	for b=1:length(batchSize_p)
		vrg{end+1} = 'sketchSize';
		vrg{end+1} = sketchSize_p(s);
		vrg{end+1} = 'batchSize';
		vrg{end+1} = batchSize_p(b);
		vrg{end+1} = 'val_size';
		vrg{end+1} = val_size;
		varargin_ = vrg;
		[resfn, dp] = demo_sketch(varargin_{:});
		r = load(resfn);
		fprintf(val_fid, 'mean mAP: %g skecthSize: %d batchSize: %d diary path: %s\n', ...
                mean(r.res(:,end)), sketchSize_p(s), batchSize_p(b), dp);
		clear r
		vrg = varargin{1};
	end
end
fclose(val_fid);
end
