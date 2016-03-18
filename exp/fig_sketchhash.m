% for getting the mAP vs. iterations figure

% for SketchHash on CIFAR 32
figname = demo_sketch('cnn', 'cifar', 32, 'noTrainingPoints', 50e3, ...
	'nbatches', 100, 'labelspercls', 0, 'override', 0, 'onlyfinal', 0, ...
	'sketchsize', 200, 'metric', 'mAP', 'ntrials', 6, 'showplots', 0)
load(figname);
[px, py] = avg_curve(res, train_iter);
px = px * 50e3/100;
figure, plot(px, py), grid
title('SketchHash CIFAR-32 50K examples, 100 batches: mAP');

% for SketchHash on Places 32
figname = demo_sketch('cnn', 'places', 32, 'noTrainingPoints', 100e3, ...
	'nbatches', 100, 'labelspercls', 0, 'override', 0, 'onlyfinal', 0, ...
	'sketchsize', 100, 'metric', 'prec_n3', 'ntrials', 3, 'showplots', 0)
load(figname);
[px, py] = avg_curve(res, train_iter);
px = px * 100e3/100;
figure, plot(px, py), grid
title('SketchHash Places-32 100K examples, 100 batches: Prec@N=3');
