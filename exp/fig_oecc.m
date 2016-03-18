% for getting the mAP vs. iterations figure


% for OECC on CIFAR 32
figname = demo_osh('cnn', 'cifar', 32, 'noTrainingPoints', 50e3, 'ntests', 20, ...
	'update_interval', 50, 'labelspercls', 0, 'override', 0, ...
	'SGDBoost', 1, 'metric', 'mAP', 'ntrials', 5, 'showplots', 1)
load(figname);
[px, py] = avg_curve(res, train_iter);
figure, plot(px, py), grid
title('OECC CIFAR-32 50K iters, U=50: mAP');

% for OECC on Places 32
demo_osh('cnn', 'places', 32, 'noTrainingPoints', 100e3, 'ntests', 20, ...
	'update_interval', 50, 'labelspercls', 0, 'override', 0, ...
	'SGDBoost', 1, 'metric', 'prec_n3', 'ntrials', 3, 'showplots', 1)
load(figname);
[px, py] = avg_curve(res, train_iter);
figure, plot(px, py), grid
title('OECC Places-32 100K iters, U=50: Prec@N=3');
