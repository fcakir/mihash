classdef MIHash
% Training routine for the MIHash method, see demo_mutualinfo.m .
%
% INPUTS
% 	Xtrain - (float) n x d matrix where n is number of points 
%       	         and d is the dimensionality 
%
% 	Ytrain - (int)   n x l matrix containing labels, for unsupervised datasets
% 			 might be empty, e.g., LabelMe.
%     thr_dist - (int)   For unlabelled datasets, corresponds to the distance 
%		         value to be used in determining whether two data instance
% 		         are neighbors. If their distance is smaller, then they are
% 		         considered neighbors.
%	       	         Given the standard setup, this threshold value
%		         is hard-wired to be compute from the 5th percentile 
% 		         distance value obtain through 2,000 training instance.
% 			 see load_gist.m . 
% 	prefix - (string) Prefix of the "checkpoint" files.
%   test_iters - (int)   A vector specifiying the checkpoints, see train.m .
%   trialNo    - (int)   Trial ID
%	opts   - (struct)Parameter structure.
%
% OUTPUTS
%  train_time  - (float) elapsed time in learning the hash mapping
%  update_time - (float) elapsed time in updating the hash table
%  res_time    - (float) elapsed time in maintaing the reservoir set
%  ht_updates  - (int)   total number of hash table updates performed
%  bit_computed_all - (int) total number of bit recomputations, see update_hash_table.m
% 
% NOTES
% 	W is d x b where d is the dimensionality 
%            and b is the bit length / # hash functions
%   Reservoir is initialized with opts.initRS instances

properties
    reservoir
end

methods
    function W = init(obj, X, opts)
        [n, d] = size(X);
        % LSH init
        W = randn(d, opts.nbits);
        W = W ./ repmat(diag(sqrt(W'*W))',d,1);

        % initialize reservoir
        % TODO hold an internal reservoir
        if reservoir_size > 0 
            ind = randperm(size(Xtrain, 1), opts.initRS);
            if ~isempty(Ytrain)
                [reservoir, update_ind] = update_reservoir(reservoir, ...
                    Xtrain(ind, :), Ytrain(ind, :), ...
                    reservoir_size, W, opts.unsupervised);
            else
                [reservoir, update_ind] = update_reservoir(reservoir, ...
                    Xtrain(ind, :), [], ...
                    reservoir_size, W, opts.unsupervised);
            end
        end
    end


    function W = train1batch(obj, W, X, Y, I, t, opts)
        ind = I(t);
        spoint = X(ind, :);
        if ~opts.unsupervised
            slabel = Y(ind, :);
        else
            slabel = [];
        end    

        % hash function update
        % TODO make mutual_info member func
        t_ = tic;
        inputs.X = spoint;
        inputs.Y = slabel;
        [obj, grad] = mutual_info(W, inputs, reservoir, ...
            opts.no_bins, opts.sigscale, opts.unsupervised, thr_dist,  1);

        % sgd
        lr = opts.stepsize * (1 ./ (1 +opts.decay *iter));
        W = W - lr * grad;
    end

end % methods

end % classdef
