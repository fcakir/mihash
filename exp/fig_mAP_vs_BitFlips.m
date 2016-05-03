% To run this script you need to initialize dataset to 'cifar' or 'places'
dataset = 'places';
dataset = 'cifar';
id = sprintf('%s-cnn-32smooth', dataset);

if ~isempty(strfind(computer, 'WIN'));
	path='\\kraken\object_detection\cachedir\online-hashing\';
else
	path = '/research/object_detection/cachedir/online-hashing/';
end
colors = {[0.85,0.325,0.098], [.5 .5 .5], [0.494, 0.184, 0.556], [0.929,0.694,0.125], [0 0.447 0.741],[0.466 0.674 0.188]};

h = figure;
set(h,'position',[200 200 720 480])
hold on;

if strcmp(dataset,'cifar')
	%----------------------CIFAR
	N = 20e3;
	U = 50;
	x_tick = (0:0.5:2)*1e4;
	x_lims = [1 1e4];
	y_lims = [0.1, 0.7];
	y_label = 'mAP';
	sketchsize = 200;

	for curve = 1:6
		if curve == 1 
			% AdaptHash
			%load([path 'adapt/' id '-A0.9B0.01S0.25-U50-50000pts-20tests\mAP_6trials.mat']);
			F = sprintf('%s/adapt/%s-A0.9B0.01S0.25-U%d-%dpts-20tests/mAP_5trials.mat', ...
				path, id, U, N);
		elseif curve == 2
			% OECC
			%load([path 'cifar-cnn-32smooth-B1S0.1-U50\50000pts_20tests\mAP_5trials.mat']);
			F = sprintf('%s/%s-B1S0.1-U%d/%dpts_20tests/mAP_5trials.mat', ...
				path, id, U, N);
		elseif curve == 3
			% OKH
			%load([path 'okh\cifar-cnn-32smooth-C1A0.1-U50-50000pts-20tests\mAP_6trials.mat']);
			F = sprintf('%s/okh/%s-C1A0.1-U%d-%dpts-20tests/mAP_5trials.mat', ...
				path, id, U, N);
		elseif curve == 4
			% SketchHash
			%F = [path 'sketch/cifar-cnn-32smooth-sketch200-50000pts-100batches/mAP_6trials.mat'];
			F = sprintf('%s/sketch/%s-Ske%dBat%d-%dpts-20tests/mAP_5trials.mat', ...
				path, id, sketchsize, U, N);
		elseif curve == 5
			% Ours
			%load([path 'cifar-cnn-32smooth-B0S0.1-U10\10000pts_10tests\mAP_5trials.mat']);
			F = sprintf('%s/%s-B0S0.1-U%d/%dpts_20tests/mAP_5trials.mat', ...
				path, id, U, N);
		else
			% Ours-dyn
			%load([path 'cifar-cnn-32smooth-B0S0.1-RS50L0.25Ada\50000pts_20tests\mAP_5trials.mat']);
			F = sprintf('%s/%s-B0S0.1-RS50L0.25Ada/%dpts_20tests/mAP_5trials.mat', ...
				path, id, N);
		end
		load(F);

		[bf, avgmAP] = avg_curve(res, bitflips);
		plot(bf, avgmAP, '-s', 'Color', colors{curve}, 'LineWidth', 2, ...
			'MarkerFaceColor', colors{curve}, 'MarkerSize', 3);
	end

elseif strcmp(dataset,'places')
	%----------------------PLACES
	N = 40e3;
	U = 100;
	x_tick = (0:4)*1e4;
	x_lims = [10 1e4];
	y_lims = [0 0.3];
	y_label = 'Precision @ N=3';
	sketchsize = 100;

	for curve = 1:6
		if curve == 1 
			% AdaptHash
			%load([path 'adapt\places-cnn-32smooth-A0.75B0.01S0.1-U50-100000pts-20tests\prec_n3_3trials.mat']);
			F = sprintf('%s/adapt/%s-A0.75B0.01S0.1-U%d-%dpts-20tests/prec_n3_3trials.mat', ...
				path, id, U, N);
		elseif curve == 2
			% OECC
			%load([path 'places-cnn-32smooth-B1S0.1-U50\100000pts_20tests\prec_n3_3trials.mat']);
			F = sprintf('%s/%s-B1S0.1-U%d/%dpts_20tests/prec_n3_3trials.mat', ...
				path, id, U, N);
		elseif curve == 3
			% OKH
			%load([path 'okh\places-cnn-32smooth-C1A0.1-U50-100000pts-20tests\prec_n3_3trials.mat']);
			F = sprintf('%s/okh/%s-C1A0.1-U%d-%dpts-20tests/prec_n3_3trials.mat', ...
				path, id, U, N);
		elseif curve == 4
			% SketchHash
			%load([path 'sketch\places-cnn-32smooth-sketch100-100000pts-100batches\prec_n3_3trials.mat']);
			F = sprintf('%s/sketch/%s-Ske%dBat%d-%dpts-20tests/prec_n3_3trials.mat', ...
				path, id, sketchsize, U, N);
		elseif curve == 5
			% Ours
			%load([path 'places-cnn-32smooth-B0S0.1-U50\100000pts_20tests\prec_n3_3trials.mat']);
			F = sprintf('%s/%s-B0S0.1-U%d/%dpts_20tests/prec_n3_3trials.mat', ...
				path, id, U, N);
		else
			% Ours-dyn
			%load([path 'places-cnn-32smooth-B0S0.1-RS50L0.01Ada\100000pts_20tests\prec_n3_3trials.mat']);
			F = sprintf('%s/%s-B0S0.1-RS50L0.01Ada/%dpts_20tests/prec_n3_3trials.mat', ...
				path, id, N);
		end
		load(F);

		[bf, avgmAP] = avg_curve(res, bitflips);
		plot(bf, avgmAP, '-s', 'Color', colors{curve}, 'LineWidth', 2, ...
			'MarkerFaceColor', colors{curve}, 'MarkerSize', 3);
	end
end

if 0
	h = legend('AdaptHash','OECC','OKH','SketchHash','Ours','Ours-dyn');
	set(h, 'Orientation', 'horizontal');
	%set(h, 'Box', 'off');
	set(h, 'fontname','Helvetica', 'fontsize', 14);
	set(h, 'TextColor',[.3 .3 .3], 'fontweight','bold');
end

set(gca,'YGrid','on');
set(gca,'gridlinestyle','-');
set(gca,'XColor', [.3 .3 .3]);
set(gca,'YColor', [.3 .3 .3]);
set(gca,'fontsize', 16);
set(gca,'fontweight', 'bold');
set(gca, 'Xscale', 'log');
ylim(y_lims);
xlim(x_lims);

x = xlabel('Bit Flips Per Example (log)');
set(x, 'fontname','Helvetica', 'fontsize',20, 'color',[.3 .3 .3], 'fontweight','bold');
set(x, 'Units', 'Normalized');
pos = get(x, 'Position');
set(x, 'Position', pos - [0, 0.02, 0]);
set(gca, 'Position', [.12 .17 .82 .8])

y = ylabel(y_label);
set(y, 'fontname','Helvetica', 'fontsize',20, 'color',[.3 .3 .3], 'fontweight','bold');
