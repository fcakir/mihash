function train_adapthash(run_trial, opts)

global Xtrain Ytrain
train_time  = zeros(1, opts.ntrials);
update_time = zeros(1, opts.ntrials);
bit_flips   = zeros(1, opts.ntrials);
parfor t = 1:opts.ntrials
    if run_trial(t) == 0
        myLogInfo('Trial %02d not required, skipped', t);
        continue;
    end
    myLogInfo('%s: %d trainPts, random trial %d', opts.identifier, opts.noTrainingPoints, t);

    % randomly set test checkpoints (to better mimic real scenarios)
    test_iters      = zeros(1, opts.ntests);
    test_iters(1)   = 1;
    test_iters(end) = opts.noTrainingPoints/2;
    interval = round(opts.noTrainingPoints/2/(opts.ntests-1));
    for i = 1:opts.ntests-2
        iter = interval*i + randi([1 round(interval/3)]) - round(interval/6);
        test_iters(i+1) = iter;
    end
    prefix = sprintf('%s/trial%d', opts.expdir, t);

    % do SGD optimization
    [train_time(t), update_time(t), bit_flips(t)] = AdaptHash(Xtrain, Ytrain, ...
        prefix, test_iters, t, opts);
end

myLogInfo('Training time (total): %.2f +/- %.2f', mean(train_time), std(train_time));
if strcmp(opts.mapping, 'smooth')
    myLogInfo('      Bit flips (per): %.4g +/- %.4g', mean(bit_flips), std(bit_flips));
end
end


% ---------------------------------------------------------
function [train_time, update_time, bitflips] = AdaptHash(...
    Xtrain, Ytrain, prefix, test_iters, trialNo, opts)
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
W = randn(d, opts.nbits);
W = W ./ repmat(diag(sqrt(W'*W))',d,1);
H = [];
% NOTE: W_lastupdate keeps track of the last W used to update the hash table
%       W_lastupdate is NOT the W from last iteration
W_lastupdate = W;
stepW = zeros(size(W));  % Gradient accumulation matrix

% set up reservoir
reservoir_size = opts.reservoirSize;
if reservoir_size > 0
    % (Xsample, Ysample): reservoir
    Xsample = zeros(reservoir_size, size(Xtrain, 2));
    Ysample = zeros(reservoir_size, 1);
    priority_queue = zeros(1, reservoir_size);
    Hres = [];  % mapped binary codes for the reservoir
else
    Xsample = []; Ysample = []; Hres = []; Hres_new = [];
end

% order training examples
if opts.pObserve > 0
    % [OPTIONAL] order training points according to label arrival strategy
    train_ind = get_ordering(trialNo, Ytrain, opts);
else
    % randomly shuffle training points before taking first noTrainingPoints
    % this fixes issue #25
    train_ind = randperm(ntrain_all, opts.noTrainingPoints);
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
%%%%%%%%%%%%%%%%%%%%%%% SET UP AdaptHash %%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%% STREAMING BEGINS! %%%%%%%%%%%%%%%%%%%%%%%
for i=1:number_iterations
    t_ = tic;

    u(1) = train_ind(2*i-1);
    u(2) = train_ind(2*i);

    sample_point1 = Xtrain(u(1),:);
    sample_point2 = Xtrain(u(2),:);
    sample_label1 = Ytrain(u(1));
    sample_label2 = Ytrain(u(2));
    s = 2*isequal(sample_label1, sample_label2);

    k_sample_data = [sample_point1;sample_point2];

    ttY = W'*k_sample_data';
    tY = single(W'*k_sample_data' > 0);
    tep = find(tY<=0);
    tY(tep) = -1;

    Dh = sum(tY(:,1) ~= tY(:,2)); 

    if s == -1     
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
    if reservoir_size > 0
        % update reservoir twice since we have a pair
        [Xsample, Ysample, priority_queue, ind] = update_reservoir(Xsample, Ysample, ...
            priority_queue, sample_point1, sample_label1, i, reservoir_size);
        [Xsample, Ysample, priority_queue, ind] = update_reservoir(Xsample, Ysample, ...
            priority_queue, sample_point2, sample_label2, i, reservoir_size);
        
        % compute new reservoir hash table (do not update yet)
        % NOTE: we always use smooth mapping for reservoir samples
        Hres_new = (W' * Xsample' > 0)';
        
        % NOTE: the old reservoir hash table needs updating too
        %       since Xsample has possibly changed.
        if isempty(Hres)
            Hres = (W_lastupdate' * Xsample' > 0)';
        elseif (ind > 0)
            Hres(ind, :) = (W_lastupdate' * Xsample(ind,:)' > 0)';
        end
    end

    % ---- determine whether to update or not ----
    [update_table, trigger_val, h_ind] = trigger_update(i, opts, ...
        W_lastupdate, W, Xsample, Ysample, Hres, Hres_new);

    % ---- update hash table ----
    if update_table
        W_lastupdate(:, h_ind) = W(:, h_ind);
        if opts.accuHash > 0
            assert(sum(sum(abs((W_lastupdate - stepW) - W))) < 1e-10);
        end
        W = W_lastupdate;
        update_iters = [update_iters, i];

        % update reservoir hash table
        if reservoir_size > 0
            Hres = Hres_new;
            if strcmpi(opts.trigger, 'bf')
                bitflips_res = bitflips_res + trigger_val;
            end
        end

        t_ = tic;
        [H, bf_all, bits_computed] = update_hash_table(H, W, Xtrain, Ytrain, ...
            multi_labeled, seenLabels, M_ecoc, opts, update_iters, h_ind);
        bits_computed_all = bits_computed_all + bits_computed;
        bitflips = bitflips + bf_all;
        update_time = update_time + toc(t_);

        %{
        Hnew = (Xtrain * W > 0)';
        if ~isempty(H)
            bitdiff = xor(H, Hnew);
            bitdiff = sum(bitdiff(:))/n;
            bitflips = bitflips + bitdiff;
            myLogInfo('[T%02d] HT update#%d @%d, bitdiff=%g', trialNo, numel(update_iters), i, bitdiff);
        else
            myLogInfo('[T%02d] HT update#%d @%d', trialNo, numel(update_iters), i);
        end
        H = Hnew;
        update_time = update_time + toc(t_);
        %}
    end

    % ---- save intermediate model ----
    if ismember(i, test_iters)
        F = sprintf('%s_iter%d.mat', prefix, i);
        save(F, 'W', 'H', 'bitflips', 'bits_computed_all', ...
            'train_time', 'update_time', 'seenLabels', 'update_iters');
        % fix permission
        if ~opts.windows, unix(['chmod g+w ' F]); unix(['chmod o-w ' F]); end

        myLogInfo('[T%02d] (%d/%d) SGD %.2fs, HTU %.2fs, %d Updates #BF=%g', ...
            trialNo, i, number_iterations, train_time, update_time, numel(update_iters), bitflips);
    end
end
%%%%%%%%%%%%%%%%%%%%%%% STREAMING ENDED! %%%%%%%%%%%%%%%%%%%%%%%

% save final model, etc
F = [prefix '.mat'];
save(F, 'W', 'H', 'bitflips', 'bits_computed_all', ...
    'train_time', 'update_time', 'test_iters', 'update_iters', ...
    'seenLabels', 'h_ind_array');
% fix permission
if ~opts.windows, unix(['chmod g+w ' F]); unix(['chmod o-w ' F]); end

ht_updates = numel(update_iters);
myLogInfo('%d Hash Table updates, bits computed: %g', ht_updates, bits_computed_all);
myLogInfo('[T%02d] Saved: %s\n', trialNo, F);
end
