function res = get_results(Htrain, Htest, Ytrain, Ytest, opts)
	% input: 
	%   Htrain - (logical) training binary codes
	%   Htest  - (logical) testing binary codes
	%   Ytrain - (int32) training labels
	%   Ytest  - (int32) testing labels
	% output:
	%  mAP - mean Average Precision

	trainsize = length(Ytrain);
	testsize  = length(Ytest);

	if strcmp(opts.metric, 'mAP')
		sim = single(2*Htrain-1)'*single(2*Htest-1);
		AP  = zeros(1, testsize);
		parfor j = 1:testsize
			labels = 2*double(Ytrain==Ytest(j))-1;
			%sim = double(2*Htrain-1)' * double(2*Htest(:, j)-1);
			[~, ~, info] = vl_pr(labels, double(sim(:, j)));
			AP(j) = info.ap;
		end
		AP = AP(~isnan(AP));  % for NUSWIDE
		res = mean(AP);
		myLogInfo(['mAP = ' num2str(res)]);


	elseif strfind(opts.metric, 'prec_k')
		% intended for PLACES, large scale
		K = opts.prec_k;
		prec_k = zeros(1, testsize);

		% distribute wrt. unique labels
		[tlabels, testind, testres, testH] = parsplit(Ytest, Htest);
		parfor l = 1:numel(testind)
			target_label = tlabels(l);
			sim = (2*single(Htrain)-1)' * (2*single(testH{l})-1);

			for i = 1:length(testind{l})
				[~, I] = sort(sim(:, i), 'descend');
				I = I(1:K);
				p_k = mean(Ytrain(I) == target_label);
				testres{l}(i) = p_k;
			end
		end
		% gather results
		for l = 1:numel(testind)
			prec_k(testind{l}) = testres{l};
		end
		res = mean(prec_k);
		myLogInfo('Prec@%d neighbors = %g', K, res);


	elseif strcmp(opts.metric, 'prec_n')
		% TODO check correctness
		N = opts.prec_n;
		R = opts.nbits;
		prec_n = zeros(1, testsize);
		sim = single(2*Htrain-1)'*single(2*Htest-1);

		for j=1:testsize
			labels = (Ytrain==Ytest(j));
			prec_n(j) = mean(labels((-sim(:,j)+R)/2 <= N));
		end
		prec_n = prec_n(~isnan(prec_n));
		res = mean(prec_n);
		myLogInfo('Prec@%d radius = %g', N, res);

	else
		error(['Evaluation metric ' opts.metric ' not implemented']);
	end
end

% ----------------------------------------------------------
function [tlabels, testind, testres, testH] = parsplit(Ytest, Htest)
	tlabels = unique(Ytest);
	testind = cell(1, length(tlabels));
	testres = cell(1, length(tlabels));
	testH = cell(1, length(tlabels));
	for l = 1:length(tlabels)
		testind{l} = find(Ytest == tlabels(l));
		testres{l} = zeros(size(testind{l}));
		testH{l} = Htest(:, testind{l});
	end
end
