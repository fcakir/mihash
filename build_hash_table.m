function Y = build_hash_table(W, traingist, trainlabels, classLabels, M, opts, H_old, h_ind)
	if strcmp(opts.mapping,'smooth')
        Y = 2*single(H_old) - 1;
		Y(h_ind,:) = 2*single(W(:,h_ind)'*traingist' > 0)-1;

	elseif strcmp(opts.mapping,'bucket')
        % For bucket mapping have to deal with the fact that
        % we may have not seen all labels, which code should we
        % use to index then? Simple solution: Use smooth mapping
        if length(classLabels) == length(unique(trainlabels))
            Y = zeros(opts.nbits, size(traingist,1), 'single');
            for i = 1:length(classLabels)
                ind = find(classLabels(i) == trainlabels);
                Y(:,ind) = repmat(M(i,:)',1,length(ind));
            end
        else
            Y = 2*single(W'*traingist' > 0)-1;
        end

	elseif strcmp(opts.mapping,'bucket2')
		Y = 2*single(W'*traingist' > 0)-1;
		sim = M * Y;
		Y = zeros(opts.nbits, size(traingist,1), 'single');
		[~, maxInd] = max(sim);
		Y = M(maxInd,:)';

    elseif strcmp(opts.mapping, 'coord')
        % KH: do extra coordinate descent step on codewords
        Y = 2*single(W'*traingist' > 0)-1;
        if length(classLabels) == length(unique(trainlabels))
            for i = 1:length(classLabels)
                ind = find(classLabels(i) == trainlabels);
                % find codeword that minimizes J
                cw = 2*single(mean(Y(:, ind), 2) > 0)-1;
                Y(:,ind) = repmat(cw, 1,length(ind));
            end
        end
    end
	% convert to logical
	Y = (Y > 0);
end
