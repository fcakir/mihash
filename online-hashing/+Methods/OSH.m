classdef OSH

% Training routine for OSH method
%
% INPUTS
% 	Xtrain - (float) n x d matrix where n is number of points 
%       	         and d is the dimensionality 
%
% 	Ytrain - (int)   n x l matrix containing labels, for unsupervised datasets
% 			 might be empty, e.g., LabelMe.
%
% NOTES
%       Adapted from original OSH implementation
% 	W is d x b where d is the dimensionality 
%       b is the bit length
%       batch size fixed to 1

properties
    ECOCs   % matrix of candidate ECOC codewords
    ECOC_i  % internal index
    ECOC_M  % assigned ECOC codewords
    seen_labels
    stepsize
    SGDBoost
end

methods
    function [W, R, obj] = init(obj, R, X, Y, opts)
        assert(~opts.unsupervised);
        assert(~isempty(Y));

        % randomly generate candidate codewords, store in ECOCs
        bigM = 10000;
        obj.ECOCs = logical(zeros(bigM, opts.nbits));
        for t = 1:opts.nbits
            r = randi([0 1], bigM, 1);
            while (sum(r)==bigM || sum(r)==0)
                r = randi([0 1], bigM, 1);
            end
            obj.ECOCs(:, t) = logical(r);
        end
        clear r
        obj.ECOC_i = 1;  
        obj.ECOC_M = zeros(0, opts.nbits);
        if size(Y, 2) > 1
            obj.seen_labels = zeros(1, size(Y, 2));
        else
            obj.seen_labels = [];
        end
        obj.stepsize = opts.stepsize;
        obj.SGDBoost = opts.SGDBoost;

        % LSH init
        d = size(X, 2);
        W = randn(d, opts.nbits);
        W = W ./ repmat(diag(sqrt(W'*W))',d,1);
    end


    function [W, ind] = train1batch(obj, W, R, X, Y, I, t, opts)
        % new training point
        ind = I(t);
        spoint = X(ind, :);
        slabel = Y(ind, :);
        
        % Assign ECOC, SGD update
        target_codes = obj.find_target_codes(slabel);
        for c = 1:size(target_codes, 1)
            code = target_codes(c, :);
            W = obj.sgd_update(W, spoint, code);
        end
    end


    function W = sgd_update(obj, W, points, codes)
        % SGD mini-batch update
        % input:
        %   W         - D*nbits matrix, each col is a hyperplane
        %   points    - n*D matrix, each row is a point
        %   codes     - n*nbits matrix, each row the corresp. target binary code
        %   stepsizes - SGD step sizes (1 per point) for current batch
        % output:
        %   updated W
        st = obj.stepsize;
        if ~obj.SGDBoost
            % no online boosting, hinge loss
            for i = 1:size(points, 1)
                xi = points(i, :);
                ci = codes(i, :);
                id = (xi * W .* ci < 1);  % logical indexing > find()
                n  = sum(id);
                if n > 0
                    W(:,id) = W(:,id) + st * (repmat(xi',[1 n])*diag(ci(id)));
                end
            end
        else
            % online boosting + exp loss
            for i = 1:size(points, 1)
                xi = points(i, :);
                ci = codes(i, :);
                for j = 1:size(W, 2)
                    if j ~= 1
                        c1 = exp(-(ci(1:j-1)*(W(:,1:j-1)'*xi')));
                    else
                        c1 = 1;
                    end
                    W(:,j) = W(:,j) - st * c1 * exp(-ci(j)*W(:,j)'*xi')*-ci(j)*xi';
                end
            end
        end
    end


    function target_codes = find_target_codes(obj, slabel)
        % find target codes for a new labeled example
        assert(~isempty(slabel) && sum(slabel) ~= 0, ...
            'Error: finding target codes for unlabeled example');

        if numel(slabel) == 1
            % single-label dataset
            [ismem, ind] = ismember(slabel, obj.seen_labels);
            if ~ismem
                ind = obj.ECOC_i;
                obj.seen_labels = [obj.seen_labels; slabel];
                obj.ECOC_M = [obj.ECOC_M; 2*obj.ECOCs(obj.ECOC_i,:)-1];
                obj.ECOC_i = obj.ECOC_i + 1;
            end
        else
            % multi-label dataset
            % find incoming labels that are unseen
            unseen = find(slabel & ~obj.seen_labels);
            if ~isempty(unseen)
                for j = unseen
                    obj.ECOC_M(j, :) = 2*obj.ECOCs(obj.ECOC_i, :)-1;
                    obj.ECOC_i = obj.ECOC_i + 1;
                end
                obj.seen_labels(unseen) = 1;
            end
            ind = find(slabel);
        end
        target_codes = obj.ECOC_M(ind, :);
    end


    function H = encode(obj, W, X, isTest)
        H = (X * W) > 0;
    end

    function P = get_params(obj)
        P = [];
    end

end % methods

end % classdef
