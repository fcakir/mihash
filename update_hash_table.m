function [Hnew, bitflips, bits_computed] = update_hash_table(H, W, Xtrain, Ytrain, ...
    h_ind, update_iters, opts, varargin)

% build new table
if opts.tstScenario == 1
    Hnew = build_hash_table(H, W, Xtrain, Ytrain, h_ind, opts, varargin{:});
else
    i = update_iters(end);
    Hnew = build_hash_table(H, W, Xtrain(1:i,:), Ytrain(1:i,:), h_ind, opts, varargin{:});
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



function H = build_hash_table(H_old, W, X, Y, h_ind, opts, varargin)
if ~isempty(varargin)
    assert(length(varargin) == 3);
    multiL = varargin{1};
    classLabels = varargin{2};
    M = varargin{3};
end

if strcmp(opts.mapping,'smooth')
    H = 2*single(H_old) - 1;
    H(h_ind,:) = 2*single(W(:,h_ind)'*X' > 0)-1;

elseif strcmp(opts.mapping,'bucket')
    % For bucket mapping have to deal with the fact that
    % we may have not seen all labels, which code should we
    % use to index then? Simple solution: Use smooth mapping
    %
    % recover true labels for single-label case
    if ~multiL, Y = floor(Y/10); end
    if length(classLabels) == length(unique(Y))
        H = zeros(opts.nbits, size(X,1), 'single');
        for i = 1:length(classLabels)
            ind = find(classLabels(i) == Y);
            H(:,ind) = repmat(M(i,:)',1,length(ind));
        end
    else
        H = 2*single(W'*X' > 0)-1;
    end

elseif strcmp(opts.mapping,'bucket2')
    H = 2*single(W'*X' > 0)-1;
    sim = M * H;
    H = zeros(opts.nbits, size(X,1), 'single');
    [~, maxInd] = max(sim);
    H = M(maxInd,:)';

elseif strcmp(opts.mapping, 'coord')
    % KH: do extra coordinate descent step on codewords
    H = 2*single(W'*X' > 0)-1;
    if length(classLabels) == length(unique(Y))
        for i = 1:length(classLabels)
            ind = find(classLabels(i) == Y);
            % find codeword that minimizes J
            cw = 2*single(mean(H(:, ind), 2) > 0)-1;
            H(:,ind) = repmat(cw, 1,length(ind));
        end
    end
end
% convert to logical
H = (H > 0);
end
