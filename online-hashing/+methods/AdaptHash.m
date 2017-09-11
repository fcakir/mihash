classdef AdaptHash

% Training routine for AdaptHash method
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
% NOTES
% 	W is d x b where d is the dimensionality 
%            and b is the bit length / # hash functions
%
% 	Data arrives in pairs. For example, if number_iterations is 1000, then 
% 	2000 points will be processed, 

properties
    alpha
    beta
    step_size
end

methods
    function [W, R, obj] = init(obj, X, R, opts)
        % alpha is the alpha in Eq. 5 in ICCV'15 paper
        % beta is the lambda in Eq. 7 in ICCV'15 paper
        % step_size is the step size of SGD
        obj.alpha       = opts.alpha;
        obj.beta        = opts.beta;
        obj.step_size   = opts.stepsize;
        disp(obj)

        % LSH init
        [n, d] = size(X);
        W = randn(d, opts.nbits);
        W = W ./ repmat(diag(sqrt(W'*W))',d,1);
    end % init


    function [W, sampleIdx] = train1batch(obj, W, R, X, Y, I, t, opts)
        [n, d] = size(X);
        sampleIdx = I(2*t-1: 2*t);
        Xsample = X(sampleIdx, :);

        ttY = W' * Xsample';
        tY  = 2 * single(ttY > 0) - 1;
        Dh  = sum(tY(:,1) ~= tY(:,2));

        if opts.unsupervised
            s = affinity(Xsample(1, :), Xsample(2, :), [], [], opts);
        else
            s = affinity([], [], Y(sampleIdx(1), :), Y(sampleIdx(2), :), opts);
        end
        s = 2 * s - 1;
        if s > 0
            loss = max(0, Dh - (1 - obj.alpha)*opts.nbits);
            ind  = find(tY(:,1) ~= tY(:,2));
            cind = find(tY(:,1) == tY(:,2));
        else
            loss = max(0, obj.alpha*opts.nbits - Dh);
            ind  = find(tY(:,1) == tY(:,2));
            cind = find(tY(:,1) ~= tY(:,2));
        end

        if loss > 0
            [ck,~] = max(abs(ttY),[],2);
            [~,ci] = sort(ck,'descend');
            ck = find(ismember(ci,ind) == 1);
            ind = ci(ck);
            ri = randperm(length(ind));
            if length(ind) > 0
                cind = [cind;ind(ceil(loss/1)+1:length(ind))];
            end

            v = W' * Xsample(1,:)'; % W' * xi
            u = W' * Xsample(2,:)'; % W' * xj

            w = (2 ./ (1 + exp(-v)) - 1) ; % f(W' * xi)
            z = (2 ./ (1 + exp(-u)) - 1) ; % f(W' * xj)

            M1 = repmat(Xsample(1,:)',1,opts.nbits);
            M2 = repmat(Xsample(2,:)',1,opts.nbits);

            t1 = exp(-v) ./ ((1 + exp(-v)).^2) ; % f'(W' * xi)
            t2 = exp(-u) ./ ((1 + exp(-u)).^2) ; % f'(W' * xj)

            D1 =  diag(2 .* z .* t1);
            D2 =  diag(2 .* w .* t2);

            M = obj.step_size * (2 * (w' * z - opts.nbits * s) * (M1 * D1 + M2 * D2));
            M(:, cind) = 0;
            M = M + obj.beta * W*(W'*W - eye(opts.nbits));
            W = W - M ;
            W = W ./ repmat(diag(sqrt(W'*W))',d,1);
        end 
    end % train1batch

end % methods

end % classdef
