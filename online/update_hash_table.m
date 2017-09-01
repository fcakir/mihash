function [Hnew, bits_computed] = update_hash_table(H, W, Xtrain, Ytrain, ...
    update_iters, opts, varargin)

% build new table
H = H_old;
H = W' * X' > 0;

bits_computed = prod(size(H));

end
