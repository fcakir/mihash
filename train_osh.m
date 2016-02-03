% ---------------------------------------------------------------------
% KH: re-organization of FC's code
%
function [W, Y] = train_osh(traingist, trainlabels, noTrainingPoints, ...
		nbits, mapping, stepsize, SGDBoost)

	% KH: randomly generate candidate codewords, store in M2
	bigM = 10000;
	M2   = zeros(bigM, nbits);
	for t = 1:nbits
		r = ones(bigM, 1);
		while (abs(sum(r)) == bigM)
			r = 2*randi([0,1], bigM, 1)-1;
		end
		M2(:, t) = r;
	end
	clear r

	% initialize with LSH
	d = size(traingist, 2);
	W = randn(d, nbits);
	W = W ./ repmat(diag(sqrt(W'*W))',d,1);

	i_ecoc = 1;
	classLabels = [];
	for i = 1:noTrainingPoints
		% sample point
		spoint = traingist(i, :);
		slabel = trainlabels(i);

		% check whether it exists in the "seen class labels" vector
		islabel = find(classLabels == slabel);
		if isempty(islabel)
			if isempty(classLabels)
				% does not exist, create a binary code for M
				classLabels = slabel;
				M = M2(i_ecoc, :);
				i_ecoc = i_ecoc + 1;
			else
				% append codeword to ECOC matrix
				classLabels = [classLabels; slabel];
				M = [M; M2(i_ecoc,:)];
				i_ecoc = i_ecoc +1;
			end
		end
		islabel = find(classLabels == slabel);

		% hash function update
		if SGDBoost == 0
			for j = 1:nbits
				if M(islabel,j)*W(:,j)'*spoint' > 1
					continue;
				else
					W(:,j) = W(:,j) + stepsize * M(islabel,j)*spoint';
				end
				%W = W ./ repmat(diag(sqrt(W'*W))',d,1);
			end
		else
			for j = 1:nbits
				if j ~= 1
					c1 = exp(-(M(islabel,1:j-1)*(W(:,1:j-1)'*spoint')));
				else
					c1 = 1;
				end
				W(:,j) = W(:,j) - stepsize * ...
					c1 * exp(-M(islabel,j)*W(:,j)'*spoint')*-M(islabel,j)*spoint';
				%W = W ./ repmat(diag(sqrt(W'*W))',d,1);
			end
		end
	end

	% populate hash table
	if strcmp(mapping,'smooth')
		Y = 2*single(W'*traingist' > 0)-1;

	elseif strcmp(mapping,'bucket')
		Y = zeros(nbits, size(traingist,1), 'single');
		for i = 1:length(classLabels)
			ind = find(classLabels(i) == trainlabels);
			Y(:,ind) = repmat(M(i,:)',1,length(ind));
		end

	elseif strcmp(mapping,'bucket2')
		Y = 2*single(W'*traingist' > 0)-1;
		sim = M * Y;
		Y = zeros(nbits, size(traingist,1), 'single');
		[~, maxInd] = max(sim);
		Y = M(maxInd,:)';

	elseif strcmp(mapping, 'coord') 
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
