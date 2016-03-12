function res = get_results(Htrain, Htest, Ytrain, Ytest, opts, cateTrainTest)
	% input: 
	%   Htrain - (logical) training binary codes
	%   Htest  - (logical) testing binary codes
	%   Ytrain - (int32) training labels
	%   Ytest  - (int32) testing labels
	% output:
	%  mAP - mean Average Precision
	if nargin < 6, cateTrainTest = []; end
	use_cateTrainTest = ~isempty(cateTrainTest);

	trainsize = length(Ytrain);
	testsize  = length(Ytest);

	if strcmp(opts.metric, 'mAP')
		sim = single(2*Htrain-1)'*single(2*Htest-1);
		%sim = Htrain'*Htest;
		AP  = zeros(1, testsize);
		parfor j = 1:testsize
			%if use_cateTrainTest
			%	labels = 2*cateTrainTest(:, j)-1;
			%else
				labels = 2*double(Ytrain==Ytest(j))-1;
			%end
			[~, ~, info] = vl_pr(labels, double(sim(:, j)));
			AP(j) = info.ap;
		end
		AP = AP(~isnan(AP));  % for NUSWIDE
		res = mean(AP);
		myLogInfo(['mAP = ' num2str(res)]);


	elseif ~isempty(strfind(opts.metric, 'prec_k'))
		% intended for PLACES, large scale
		K = opts.prec_k;
		prec_k = zeros(1, testsize);
		sim = single(2*Htrain-1)'*single(2*Htest-1);
		%sim = Htrain'*Htest;

		parfor i = 1:testsize
			%if use_cateTrainTest 
			%	labels = cateTrainTest(:, i);
			%else
				labels = (Ytrain == Ytest(i));
			%end
			sim_i = sim(:, i);
			%th = binsearch(sim_i, K);
			%[~, I] = sort(sim_i(sim_i>th), 'descend');
			[~, I] = sort(sim_i, 'descend');
			I = I(1:K);
			prec_k(i) = mean(labels(I));
		end
		%prec_k = prec_k(~isnan(prec_k));
		res = mean(prec_k);
		myLogInfo('Prec@(neighs=%d) = %g', K, res);


	elseif ~isempty(strfind(opts.metric, 'prec_n'))
		N = opts.prec_n;
		R = opts.nbits;
		prec_n = zeros(1, testsize);
		sim = single(2*Htrain-1)'*single(2*Htest-1);
		%sim = Htrain'*Htest;

		% NOTE 'for' has better CPU usage
		for j=1:testsize
			if use_cateTrainTest
				labels = cateTrainTest(:, j);
			else
				labels = (Ytrain == Ytest(j));
			end
			ind = find(R-sim(:,j) <= 2*N);
			if ~isempty(ind)
				prec_n(j) = mean(labels(ind));
			end
		end
		%prec_n = prec_n(~isnan(prec_n));
		res = mean(prec_n);
		myLogInfo('Prec@(radius=%d) = %g', N, res);

	else
		error(['Evaluation metric ' opts.metric ' not implemented']);
	end
end

% ----------------------------------------------------------
function T = binsearch(x, k)
	% x: input vector
	% k: number of largest elements
	% T: threshold
	T = -Inf;
	while numel(x) > k
		T0 = T;
		x0 = x;
		T  = mean(x);
		x  = x(x>T);
	end
	% for sanity
	if numel(x) < k, T = T0; end
end
