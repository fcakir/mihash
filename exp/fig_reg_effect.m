% To run this script you need to initialize dataset to 'cifar' or 'places'
dataset = 'cifar';
dataset = 'places';

use_avg_curve = false;

if ~isempty(strfind(computer, 'WIN'));
	path='\\kraken\object_detection\cachedir\online-hashing\';
else
	path = '/research/object_detection/cachedir/online-hashing/';
end

lambda = [0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 50, 100];
if strcmp(dataset, 'places')
	lambda = [lambda, 200, 500, 1000];
	T = 3;
	U = 100;
	N = 20e3;
	metric = 'prec_n3';
	clrind = 1;
	y_lims = [5.1, 11]*1e-4;
	y_tick = (6:11)*1e-4;
else
	lambda = [0.001, lambda];
	T = 10;
	U = 50;
	N = 10e3;
	metric = 'mAP';
	clrind = 2;
	y_lims = [5.1, 6.5]*1e-3;
	y_tick = (5.3:0.2:6.5)*1e-3;
end
colors = { [0.466 0.674 0.188], [0 0.447 0.741],[0.85,0.325,0.098], [0.929,0.694,0.125], [0.494, 0.184, 0.556]}; 
% , [0.301,0.745,0.933], [0.635, 0.078, 0.184],[0.95,0.95,0],[0.8,0,0.8]};

h = figure;
set(h,'position',[100 100 650 390])
hold on;


load(sprintf('%s/%s-cnn-32smooth-B0S0.1-U%d/%dpts_20tests/%s_%dtrials.mat', ...
	path, dataset, U, N, metric, T)); 

if use_avg_curve 
	[train_iter, res] = avg_curve(res, train_iter);
	Area = train_iter(end) - train_iter(1);
	avgmAP = res/Area;
	baseline = trapz(train_iter, avgmAP) / mean(bitflips(:, end));
else
	baseline = zeros(1, size(res, 1));
	for i = 1:size(res, 1)
		Area = train_iter(i, end) - train_iter(i, 1);
		avgmAP = res(i, :)/Area;
		baseline(i) = trapz(train_iter(i, :), avgmAP) / bitflips(i, end);
	end
	baseline = mean(baseline);
end
baseline
baseline_bf = mean(bitflips(:, end))

hline = plot([min(lambda), max(lambda)], [baseline, baseline]);
set(hline,'LineStyle','--','LineWidth',3,'Color',colors{clrind});


result = [];
result_bf = [];
for i = 1:length(lambda)
	load(sprintf('%s/%s-cnn-32smooth-B0S0.1-RS50L%gU%d/%dpts_20tests/%s_%dtrials.mat', ...
		path, dataset, lambda(i), U, N, metric, T)); 
	if use_avg_curve 
		[train_iter, res] = avg_curve(res, train_iter);
		Area = train_iter(end) - train_iter(1);
		avgmAP = res/Area;
		reg = trapz(train_iter, avgmAP) / mean(bitflips(:, end));
	else
		reg = zeros(1, size(res, 1));
		for i = 1:size(res, 1)
			Area = train_iter(i, end) - train_iter(i, 1);
			avgmAP = res(i, :)/Area;
			reg(i) = trapz(train_iter(i, :), avgmAP) / bitflips(i, end);
		end
		reg = mean(reg);
	end

	result = [result, reg];
	result_bf = [result_bf, mean(bitflips(:, end))];
end
result
result_bf

p = plot(lambda, result, '-s');
set(p, 'Color', colors{clrind}, 'LineWidth', 3, 'MarkerFaceColor', colors{clrind});



h=legend('Ours (\lambda=0)', 'Ours + \lambda');
set(h,'fontname','Helvetica','fontsize',16,'TextColor',[.3 .3 .3],'fontweight','bold');

ylim(y_lims);
xlim([0, max(lambda)]);
set(gca,'fontsize', 16);
%set(gca, 'fontweight', 'bold');
set(gca,'YGrid','on');
set(gca,'gridlinestyle','-');
set(gca,'XColor', [.3 .3 .3]);
set(gca,'YColor', [.3 .3 .3]);
set(gca,'Xscale','log');
set(gca, 'Ytick', y_tick);

x = xlabel('Regularization Parameter \lambda');
%set(x,'fontname','Helvetica','fontsize',20,'color',[.3 .3 .3],'fontweight','bold');
%set(x, 'Units', 'Normalized');
%pos = get(x, 'Position');
%set(x, 'Position', pos - [0, 0.01, 0]);

y = ylabel('AUC / BitFlips');
%set(y, 'fontname','Helvetica','fontsize',20,'color',[.3 .3 .3],'fontweight','bold');
