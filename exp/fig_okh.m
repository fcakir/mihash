% for getting the mAP vs. iterations figure

% best parameters indicated by the paper: (0.1, 1)
% seemed to be the most robust combination

% for OKH on CIFAR 32
figname = demo_okh('cnn', 'cifar', 32, 'noTrainingPoints', 50e3, 'ntests', 20, ...
	'updateInterval', 50, 'labelspercls', 0, 'override', 0, ...
	'alpha', 0.1, 'c', 1, 'metric', 'mAP', 'ntrials', 6, 'showplots', 0)
load(figname);
[px, py] = avg_curve(res, train_iter);
px = px * 2;  % OKH uses pairs
figure, plot(px, py), grid
title('OKH CIFAR-32 50K iters, U=50: mAP');
pause

% for OKH on Places 32
figname = demo_okh('cnn', 'places', 32, 'noTrainingPoints', 100e3, 'ntests', 20, ...
	'updateInterval', 50, 'labelspercls', 0, 'override', 0, ...
	'alpha', 0.1, 'c', 1, 'metric', 'prec_n3', 'ntrials', 3, 'showplots', 0)
load(figname);
[px, py] = avg_curve(res, train_iter);
px = px * 2;  % OKH uses pairs
figure, plot(px, py), grid
title('OKH Places-32 100K iters, U=50: Prec@N=3');
