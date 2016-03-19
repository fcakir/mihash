function demo_retrieval(dataset)

	% assuming CNN features and 32 bits
	if strcmp(dataset, 'cifar')
		opts = get_opts('cnn', dataset, 32, 'ntrials', 1, 'ntests', 2, ...
			'noTrainingPoints', 50e3, 'update_interval', 50e3, 'showplots', 0, ...
			'nworkers', 0);
		visFunc = @vis_cifar;
	else
		opts = get_opts('cnn', 'places', 32, 'ntrials', 1, 'ntests', 2, ...
			'noTrainingPoints', 50e3, 'update_interval', 50e3, 'showplots', 0, ...
			'labelspercls', 0, 'metric', 'prec_n3', 'nworkers', 0);
		visFunc = @vis_places;
	end


	% load data (only if necessary)
	% Names: struct of two fields, 'train' & 'test'
	global Xtrain Xtest Ytrain Ytest Names Dtype
	Dtype_this = [dataset '_' opts.ftype];
	if ~isempty(Dtype) && strcmp(Dtype_this, Dtype)
		myLogInfo('Dataset already loaded for %s', Dtype_this);
	else
		myLogInfo('Loading data for %s...', Dtype_this);
		eval(['[Xtrain, Ytrain, Xtest, Ytest, Names] = load_', opts.ftype, ...
			'(dataset, opts);']);
		Dtype = Dtype_this;
	end


	% run training
	modelfn = sprintf('%s/trial1.mat', opts.expdir);
	if ~exist(modelfn, 'file')
		myLogInfo('Training model...');
		run_trial = 1;
		train_osh(run_trial, opts);
	else
		myLogInfo('Already trained');
	end


	% get results
	d = load(modelfn);
	Htrain = d.H;
	Htest  = (d.W' * Xtest' > 0);
	sim = (2*Htrain-1)'*(2*Htest-1);


	% visualize
	if ~exist('res', 'dir'), mkdir('res'); end
	visFunc(sim, floor(Ytrain/10), floor(Ytest/10), Names);

end


% ---------------------------------------------------------------
function vis_places(sim, trainY, testY, Names)

	% sample some test images
	K = 20; %(?)
	for i = randperm(length(testY), K)
		name = Names.test{i};
		fn = sprintf('res/query%d.png', i);
		try
			web_save(name, fn);
		catch, name, continue
		end
		myLogInfo('saved %s', fn);

		% get top 10 results
		myLogInfo('Label %d, top 10 retrieved labels:', testY(i));
		[val, ind] = sort(sim(:, i), 'descend');

		for j = 1:10
			fprintf('%d ', trainY(ind(j)));
			name = Names.train{ind(j)};
			fn = sprintf('res/query%d_retrieval%02d.png', i, j);
			try
				web_save(name, fn);
			catch, name, continue;
			end
		end
		fprintf('\n\n');
	end
end


function web_save(name, fn)
	prefix = 'http://monday.csail.mit.edu/cmr/RGB';

	[path, imfn, ext] = fileparts(name);  % get image name
	[path, cls, ~] = fileparts(path);  % get class name
	[path, c, ~] = fileparts(path);  % get preceeding string
	if length(c) == 1  % single character
		url = sprintf('%s/%s/%s/%s', prefix, c, cls, [imfn ext]);
	else
		supercls = c;
		[path, c, ~] = fileparts(path);
		url = sprintf('%s/%s/%s/%s/%s', prefix, c, supercls, cls, [imfn ext]);
	end

	urlwrite(url, fn);
end

% ---------------------------------------------------------------
function vis_cifar(sim, trainY, testY, Names)
	% sample one(?) test image from each class 
	K = 2;
	for y = 1:10
		ind = find(testY == y);
		ind = ind(randperm(length(ind), K));

		for i = ind'
			myLogInfo('Label %d, top 10 retrieved labels:', testY(i));

			% get top 10 results
			[val, ind] = sort(sim(:, i), 'descend');

			for j = 1:10
				fprintf('%d ', trainY(ind(j)));
				im = somehow_get_image( Names.train(ind(j)) );
				fn = sprintf('query%d_retrieval%d.png', i, j);
				imwrite(im, fn);
			end
			fprintf('\n\n');
		end
	end
end
