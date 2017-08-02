function [Hnew, bitflips, bits_computed] = update_hash_table(H, W, Xtrain, Ytrain, ...
    h_ind, update_iters, opts, varargin)

% build new table
H = H_old;
H(h_ind, :) = W(:, h_ind)' * X' > 0;

% compute bitflips
if isempty(H)
    bitflips = 0;
    bits_computed = length(h_ind) * size(Hnew, 2); % if H is empty, length(h_ind) should be nbits
else
    bitdiff = xor(H, Hnew);
    bitflips = sum(bitdiff(:))/size(Xtrain, 1);
    bits_computed = length(h_ind)*size(Hnew, 2);
end
end
