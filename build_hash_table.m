function Y = build_hash_table(W, traingist, trainlabels, classLabels, M, opts)
	if strcmp(opts.mapping,'smooth')
		Y = 2*single(W'*traingist' > 0)-1;

	elseif strcmp(opts.mapping,'bucket')
		Y = zeros(nbits, size(traingist,1), 'single');
		for i = 1:length(classLabels)
			ind = find(classLabels(i) == trainlabels);
			Y(:,ind) = repmat(M(i,:)',1,length(ind));
		end

	elseif strcmp(opts.mapping,'bucket2')
		Y = 2*single(W'*traingist' > 0)-1;
		sim = M * Y;
		Y = zeros(nbits, size(traingist,1), 'single');
		[~, maxInd] = max(sim);
		Y = M(maxInd,:)';

	elseif strcmp(opts.mapping, 'coord') 
		% KH: do extra coordinate descent step on codewords
		Y = 2*single(W'*traingist' > 0)-1;
		for i = 1:length(classLabels)
			ind = find(classLabels(i) == trainlabels);
			% find codeword that minimizes J
			cw = 2*single(mean(Y(:, ind), 2) > 0)-1; 
			Y(:,ind) = repmat(cw, 1,length(ind));
		end
	end
end
