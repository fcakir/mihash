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
        code_length
    end

    function init(obj, X, opts)
        % alphaa is the alpha in Eq. 5 in the ICCV paper
        % beta is the lambda in Eq. 7 in the ICCV paper
        % step_size is the step_size of the SGD
        alphaa      = opts.alpha; %0.8;
        beta        = opts.beta; %1e-2;
        step_size   = opts.stepsize; %1e-3;
        [n, d]      = size(X);
        code_length = opts.nbits;
        number_iterations = opts.noTrainingPoints/2;
        logInfo('[T%02d] %d training iterations', trialNo, number_iterations);

        % LSH init
        W = randn(d, opts.nbits);
        W = W ./ repmat(diag(sqrt(W'*W))',d,1);
    end

    function train1batch(obj, W, X, Y, I, t, opts)
        u = I(2*t-1: 2*t);
        sample_point1 = X(u(1),:);
        sample_point2 = X(u(2),:);

        if ~opts.unsupervised
            sample_label1 = Y(u(1));
            sample_label2 = Y(u(2));
            s = 2*isequal(sample_label1, sample_label2)-1;
        else
            sample_label1 = [];sample_label2 = [];
            s = 2*(pdist([sample_point1;sample_point2],'euclidean') <= thr_dist) - 1;
        end

        k_sample_data = [sample_point1; sample_point2];

        ttY = W' * k_sample_data';
        tY  = single(W' * k_sample_data' > 0);
        tY(tY <= 0) = -1;

        Dh = sum(tY(:,1) ~= tY(:,2)); 

        if s <= 0
            loss = max(0, alphaa*code_length - Dh);
            ind  = find(tY(:,1) == tY(:,2));
            cind = find(tY(:,1) ~= tY(:,2));
        else
            loss = max(0, Dh - (1 - alphaa)*code_length);
            ind  = find(tY(:,1) ~= tY(:,2));
            cind = find(tY(:,1) == tY(:,2));
        end


        if ceil(loss) ~= 0
            [ck,~] = max(abs(ttY),[],2);
            [~,ci] = sort(ck,'descend');
            ck = find(ismember(ci,ind) == 1);
            ind = ci(ck);
            ri = randperm(length(ind));
            if length(ind) > 0
                cind = [cind;ind(ceil(loss/1)+1:length(ind))];
            end

            v = W' * k_sample_data(1,:)'; % W' * xi
            u = W' * k_sample_data(2,:)'; % W' * xj

            w = (2 ./ (1 + exp(-v)) - 1) ; % f(W' * xi)
            z = (2 ./ (1 + exp(-u)) - 1) ; % f(W' * xj)

            M1 = repmat(k_sample_data(1,:)',1,code_length);
            M2 = repmat(k_sample_data(2,:)',1,code_length);

            t1 = exp(-v) ./ ((1 + exp(-v)).^2) ; % f'(W' * xi)
            t2 = exp(-u) ./ ((1 + exp(-u)).^2) ; % f'(W' * xj)

            D1 =  diag(2 .* z .* t1);
            D2 =  diag(2 .* w .* t2);

            M = step_size * (2 * (w' * z - code_length * s) * (M1 * D1 + M2 * D2));

            M(:,cind) = 0;
            M = M + beta * W*(W'*W - eye(code_length));
            W = W - M ;
            W = W ./ repmat(diag(sqrt(W'*W))',d,1);
        end 
    end
end
