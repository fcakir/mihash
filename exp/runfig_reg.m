function runfig_reg(dataset, lambdas, varargin)

	if strcmp(dataset, 'cifar')
		ntrain = 10e3;
		metric = 'mAP';
		U = 50;
		T = 5;
	else
		assert(strcmp(dataset, 'places'));
		ntrain = 20e3;
		metric = 'prec_n3';
		U = 100;
		T = 3;
	end

	for L = lambdas
		demo_osh('cnn', dataset, 32, 'noTrainingPoints', ntrain, 'ntests', 20, ...
			'ntrials', T, 'updateInterval', U, 'reg_rs', L, 'metric', metric, ...
			'showplots', 0, varargin{:})
	end

end
