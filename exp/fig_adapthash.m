% for getting the mAP vs. iterations figure

% for AdaptHash on CIFAR 32
figname = demo_adapthash('cnn', 'cifar', 32, 'noTrainingPoints', 50e3, 'ntests', 20, ...
	'update_interval', 50, 'labelspercls', 0, 'override', 0, ...
	'alpha', 0.9, 'stepsize', 0.25, 'metric', 'mAP', 'ntrials', 6, 'showplots', 1)
load(figname);
[px, py] = avg_curve(res, train_iter);
px = px * 2;  % AdaptHash uses pairs
figure, plot(px, py), grid
title('AdaptHash CIFAR-32 50K iters, U=50: mAP');

% for AdaptHash on Places 32
figname = demo_adapthash('cnn', 'places', 32, 'noTrainingPoints', 100e3, 'ntests', 20, ...
	'update_interval', 50, 'labelspercls', 0, 'override', 0, ...
	'alpha', 0.75, 'stepsize', 0.1, 'metric', 'prec_n3', 'ntrials', 3, 'showplots', 1)
load(figname);
[px, py] = avg_curve(res, train_iter);
px = px * 2;  % AdaptHash uses pairs
figure, plot(px, py), grid
title('AdaptHash Places-32 100K iters, U=50: Prec@N=3');
