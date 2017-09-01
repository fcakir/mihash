function A = affinity(X, Y, opts)
% compute binary affinity matrix
if opts.unsupervised || isempty(Y)
    A = pdist(X, 'Euclidean') < opts.thr_dist;
else
    A = bsxfun(@eq, Y, Y');
end
end
