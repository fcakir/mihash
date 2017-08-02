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


% ------------------ helper function -------------------------------
function H = build_hash_table(H_old, W, X, Y, h_ind, opts, varargin)
H = H_old;
H(h_ind, :) = W(:, h_ind)' * X' > 0;
end
