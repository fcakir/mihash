classdef OKH
% Training routine for OKH method
%
% INPUTS
% 	Xtrain - (float) n x d matrix where n is number of points 
%       	         and d is the dimensionality 
%
% 	Ytrain - (int)   n x l matrix containing labels, for unsupervised datasets
% 			 might be empty, e.g., LabelMe.
%
% NOTES
%       Adapted from original OKH implementation
% 	W is d x b where d is the kernel mapping dimensionality (default 300+1)
%       and b is the bit length
%
% 	data arrives in pairs. If number_iterations is 1000, then 
% 	2000 points will be processed

properties
    para
    sigma
end

properties (Transient = true)
    KX
end

methods
    function [W, R, obj] = init(obj, R, X, Y, opts)
        % do kernel mapping to Xtrain
        % KX: each COLUMN is a kernel-mapped training example
        assert(size(X, 1) >= 4000);

        % sample support samples (300) from the FIRST HALF of training set
        nhalf = floor(size(X, 1)/2);
        ind = randperm(nhalf, 300);
        Xanchor = X(ind, :);
        logInfo('Randomly selected 300 anchor points');

        % estimate sigma for Gaussian kernel using samples from the SECOND HALF
        ind = randperm(nhalf, 2000);
        Xval = X(nhalf+ind, :);
        Kval = sqdist(Xval', Xanchor');
        obj.sigma = mean(mean(Kval, 2));
        logInfo('Estimated sigma = %g', obj.sigma);
        clear Xval Kval

        % kernel mapping the whole set
        obj.KX = exp(-0.5*sqdist(X', Xanchor')/obj.sigma^2)';
        obj.KX = [obj.KX; ones(1,size(obj.KX,2))];

        obj.para = [];
        obj.para.c      = opts.c; %0.1;
        obj.para.alpha  = opts.alpha; %0.2;
        obj.para.anchor = Xanchor;
        disp(obj)

        % LSH init
        d = size(obj.KX, 1);
        W = randn(d, opts.nbits);
        W = W ./ repmat(diag(sqrt(W'*W))',d,1);
    end


    function [W, ind] = train1batch(obj, W, R, X, Y, I, t, opts)
        ind = I(2*t-1 : 2*t);
        if opts.unsupervised
            Y1 = [];
            Y2 = [];
        else
            Y1 = Y(ind(1), :);
            Y2 = Y(ind(2), :);
        end
        s = affinity(X(ind(1), :), X(ind(2), :), Y1, Y2, opts);
        s = 2 * s - 1;

        % hash function update
        xi = obj.KX(:, ind(1));
        xj = obj.KX(:, ind(2));
        W  = Methods.OKHlearn(xi, xj, s, W, obj.para);
    end


    function H = encode(obj, W, X, isTest)
        if isTest
            % do kernel mapping for test data
            X = exp(-0.5*sqdist(X', obj.para.anchor')/obj.sigma^2)';
            X = [X; ones(1, size(X,2))];
            H = (X' * W) > 0;
        else
            H = (obj.KX' * W) > 0;
        end
    end

    function P = get_params(obj)
        P = [];
        P.sigma = obj.sigma;
        P.Xanchor = obj.para.anchor;
    end

end % methods

end % classdef
