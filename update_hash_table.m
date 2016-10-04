function [Hnew, bitflips, bits_computed] = update_hash_table(H, W, Xtrain, Ytrain, ...
    multi_labeled, seenLabels, M_ecoc, opts, update_iters, h_ind)
% do actual hash table update

% recover true labels for single-label case
if ~multi_labeled, Ytrain = floor(Ytrain/10); end

% build new table
if opts.tstScenario == 1
    Hnew = build_hash_table(W, Xtrain, Ytrain, seenLabels, M_ecoc, opts, H, h_ind);
else
    i = update_iters(end);
    Hnew = build_hash_table(W, Xtrain(1:i,:), Ytrain(1:i,:), seenLabels, M_ecoc, opts, H, h_ind);
end

% compute bitflips
if isempty(H)
    bitflips = 0;
    if opts.tstScenario == 1
        bits_computed = length(h_ind) * size(Hnew, 2); % if H is empty, length(h_ind) should be nbits
    else
        bits_computed = length(h_ind) * update_iters(end-1);
    end
else
    if opts.tstScenario == 2
        bitdiff = xor(H, Hnew(:, 1:update_iters(end-1)));
        bitflips = sum(bitdiff(:))/update_iters(end-1);
        bits_computed = length(h_ind)*update_iters(end-1);
    else
        bitdiff = xor(H, Hnew);
        bitflips = sum(bitdiff(:))/size(Xtrain, 1);
        bits_computed = length(h_ind)*size(Hnew, 2);
    end
end
end



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
