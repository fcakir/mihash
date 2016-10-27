function [train_time, update_time, res_time, ht_updates, bits_computed_all, bitflips] = ...
    train_adapthash(Xtrain, Ytrain, thr_dist, prefix, test_iters, trialNo, opts)
% Xtrain (float) n x d matrix where n is number of points 
%                   and d is the dimensionality 
%
% Ytrain (int) is n x 1 matrix containing labels 
%
% W is d x b where d is the dimensionality 
%            and b is the bit length / # hash functions
%
% if number_iterations is 1000, this means 2000 points will be processed, 
% data arrives in pairs


[n,d]       = size(Xtrain);
tu          = randperm(n);

% alphaa is the alpha in Eq. 5 in the ICCV paper
% beta is the lambda in Eq. 7 in the ICCV paper
% step_size is the step_size of the SGD
alphaa      = opts.alpha; %0.8;
beta        = opts.beta; %1e-2;
step_size   = opts.stepsize; %1e-3;


%%%%%%%%%%%%%%%%%%%%%%% INIT %%%%%%%%%%%%%%%%%%%%%%%
if 0
    W = rand(d, opts.nbits)-0.5;
else
    % LSH init
    W = randn(d, opts.nbits);
    W = W ./ repmat(diag(sqrt(W'*W))',d,1);
end
H = [];
% NOTE: W_lastupdate keeps track of the last W used to update the hash table
%       W_lastupdate is NOT the W from last iteration
W_lastupdate = W;
stepW = zeros(size(W));  % Gradient accumulation matrix

% are we handling a mult-labeled dataset?
multi_labeled = (size(Ytrain, 2) > 1);
if multi_labeled, myLogInfo('Handling multi-labeled dataset'); end

% set up reservoir
reservoir = [];
reservoir_size = opts.reservoirSize;
if reservoir_size > 0
    reservoir.size = 0;
    reservoir.X    = zeros(0, size(Xtrain, 2));
    reservoir.PQ   = [];
    reservoir.H    = [];  % mapped binary codes for the reservoir
    if opts.unsupervised
	reservoir.Y = [];
    else
        reservoir.Y  = zeros(0, size(Ytrain, 2));
    end
end

% order training examples
if opts.pObserve > 0
    % [OPTIONAL] order training points according to label arrival strategy
    train_ind = get_ordering(trialNo, Ytrain, opts);
else
    % randomly shuffle training points before taking first noTrainingPoints
    train_ind = randperm(size(Xtrain, 1), opts.noTrainingPoints);
end
%%%%%%%%%%%%%%%%%%%%%%% INIT %%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%% SET UP AdaptHash %%%%%%%%%%%%%%%%%%%%%%%
% for AdaptHash
code_length = opts.nbits;
number_iterations = opts.noTrainingPoints/2;
myLogInfo('[T%02d] %d training iterations', trialNo, number_iterations);

% bit flips & bits computed
bitflips          = 0;
bitflips_res      = 0;
bits_computed_all = 0;

% HT updates
update_iters = [];
h_ind_array  = [];

% for recording time
train_time  = 0;  
update_time = 0;
res_time    = 0;
%%%%%%%%%%%%%%%%%%%%%%% SET UP AdaptHash %%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%% STREAMING BEGINS! %%%%%%%%%%%%%%%%%%%%%%%
for iter = 1:number_iterations
    t_ = tic;

    u(1) = train_ind(2*iter-1);
    u(2) = train_ind(2*iter);

    sample_point1 = Xtrain(u(1),:);
    sample_point2 = Xtrain(u(2),:);
    if ~opts.unsupervised
        sample_label1 = Ytrain(u(1));
        sample_label2 = Ytrain(u(2));
        s = 2*isequal(sample_label1, sample_label2)-1;
    else
	sample_label1 = [];sample_label2 = [];
	s = 2*(pdist([sample_point1;sample_point2],'euclidean') <= thr_dist) - 1;
    end

    k_sample_data = [sample_point1;sample_point2];

    ttY = W'*k_sample_data';
    tY = single(W'*k_sample_data' > 0);
    tep = find(tY<=0);
    tY(tep) = -1;

    Dh = sum(tY(:,1) ~= tY(:,2)); 

    %if s == -1     
    if s <= 0  % KH: safety precaution
        loss = max(0, alphaa*code_length - Dh);
        ind = find(tY(:,1) == tY(:,2));
        cind = find(tY(:,1) ~= tY(:,2));
    else
        loss = max(0, Dh - (1 - alphaa)*code_length);
        ind = find(tY(:,1) ~= tY(:,2));
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

    train_time = train_time + toc(t_);

    % ---- reservoir update & compute new reservoir hash table ----
    t_ = tic;
    Hres_new = [];
    if reservoir_size > 0
        [reservoir, update_ind] = update_reservoir(reservoir, k_sample_data, ...
            [sample_label1; sample_label2], reservoir_size, W_lastupdate, opts.unsupervised);
        % compute new reservoir hash table (do not update yet)
        Hres_new = (W' * reservoir.X' > 0)';
    end

    % ---- determine whether to update or not ----
    [update_table, trigger_val, h_ind] = trigger_update(iter, ...
        opts, W_lastupdate, W, reservoir, Hres_new, opts.unsupervised, thr_dist);
    inv_h_ind = setdiff(1:opts.nbits, h_ind);  % keep these bits unchanged
    if reservoir_size > 0 && numel(h_ind) < opts.nbits  % selective update
        assert(opts.fracHash < 1);
        Hres_new(:, inv_h_ind) = reservoir.H(:, inv_h_ind);
    end
    res_time = res_time + toc(t_);

    % ---- hash table update, etc ----
    if update_table
        h_ind_array = [h_ind_array; single(ismember(1:opts.nbits, h_ind))];
        W_lastupdate(:, h_ind) = W(:, h_ind);
        if opts.accuHash <= 0
            W = W_lastupdate;
            myLogInfo('not accumulating gradients!');
        end
        if opts.fracHash < 1
            myLogInfo('selective update: fracHash=%g, randomHash=%g', ...
                opts.fracHash, opts.randomHash);
        end
        update_iters = [update_iters, iter];

        % update reservoir hash table
        if reservoir_size > 0
            reservoir.H = Hres_new;
            if strcmpi(opts.trigger, 'bf')
                bitflips_res = bitflips_res + trigger_val;
            end
        end

        % actual hash table update (record time)
        t_ = tic;
        [H, bf_all, bits_computed] = update_hash_table(H, W_lastupdate, ...
            Xtrain, Ytrain, h_ind, update_iters, opts);
        bits_computed_all = bits_computed_all + bits_computed;
        bitflips = bitflips + bf_all;
        update_time = update_time + toc(t_);
    end

    % ---- save intermediate model ----
    if ismember(iter, test_iters)
        F = sprintf('%s_iter%d.mat', prefix, iter);
        save(F, 'W', 'W_lastupdate', 'H', 'bitflips', 'bits_computed_all', ...
            'train_time', 'update_time', 'res_time', 'update_iters');
        % fix permission
        if ~opts.windows, unix(['chmod g+w ' F]); unix(['chmod o-w ' F]); end

        myLogInfo(['[T%02d] %s\n' ...
            '     (%d/%d) W %.2fs, HT %.2fs(%d updates), Res %.2fs\n' ...
            '     total #BRs=%g, avg #BF=%g'], ...
            trialNo, opts.identifier, iter*opts.batchSize, opts.noTrainingPoints, ...
            train_time, update_time, numel(update_iters), res_time, ...
            bits_computed_all, bitflips);
    end
end
%%%%%%%%%%%%%%%%%%%%%%% STREAMING ENDED! %%%%%%%%%%%%%%%%%%%%%%%

% save final model, etc
F = [prefix '.mat'];
save(F, 'W', 'H', 'bitflips', 'bits_computed_all', ...
    'train_time', 'update_time', 'res_time', 'test_iters', 'update_iters', ...
    'h_ind_array');
% fix permission
if ~opts.windows, unix(['chmod g+w ' F]); unix(['chmod o-w ' F]); end

ht_updates = numel(update_iters);
myLogInfo('%d Hash Table updates, bits computed: %g', ht_updates, bits_computed_all);
myLogInfo('[T%02d] Saved: %s\n', trialNo, F);
end
