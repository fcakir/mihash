dataset = {'cifar', 'places'};
ntrain  = [50e3, 100e3];

lambdas = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2];

for i = 1:2
	figure

	% baseline (loss only), U=50
	f = demo_osh('cnn', dataset{i}, 32, 'noTrainingPoints', ntrain(i), 'update_interval', 50, ...
		'ntests', 20);
	d = load(f);
	[px, py] = avg_curve(d.res, d.train_iters);
	plot()

	
	for l = lambdas
		% loss + reg_rs, U=50 
		f = demo_osh('cnn', dataset{i}, 32, 'noTrainingPoints', ntrain(i), 'update_interval', 50, ...
			'ntests', 20, 'reg_rs', l, 'samplesize', 50);
		
		% loss + reg_rs, adaptive
		f = demo_osh('cnn', dataset{i}, 32, 'noTrainingPoints', ntrain(i), 'adaptive', 1, ...
			'ntests', 20, 'reg_rs', l, 'samplesize', 50);
	end
end
