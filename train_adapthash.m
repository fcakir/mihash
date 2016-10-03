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


% KH
W = randn(d, opts.nbits);
W = W ./ repmat(diag(sqrt(W'*W))',d,1);
H = [];
code_length = opts.nbits;
number_iterations = opts.noTrainingPoints/2;
myLogInfo('[T%02d] %d training iterations', trialNo, number_iterations);
bitflips = 0;
train_time = 0;
update_time = 0;
update_iters = [];
% order training examples
if opts.pObserve > 0
    % [OPTIONAL] order training points according to label arrival strategy
    train_ind = get_ordering(trialNo, Ytrain, opts);
else
    % randomly shuffle training points before taking first noTrainingPoints
    % this fixes issue #25
    train_ind = randperm(ntrain_all, opts.noTrainingPoints);
end
% KH

for i=1:number_iterations

    t_ = tic;

    u(1) = train_ind(2*i-1);
    u(2) = train_ind(2*i);

    sample_point1 = Xtrain(u(1),:);
    sample_point2 = Xtrain(u(2),:);
    s = 2*isequal(Ytrain(u(1)), Ytrain(u(2)))-1;

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

    % determine whether to update or not
    [update_table, trigger_val, h_ind] = trigger_update(i, opts, ...
        W_lastupdate, W, Xsample, Ysample, Hres, Hres_new);

    update_table = false;
    if i == 1 || i == number_iterations
        update_table = true;
    elseif (opts.updateInterval == 2 && ismember(i, test_iters)) || ...
            (opts.updateInterval > 2 && ~mod(i, opts.updateInterval/2))
        update_table = true;
    end


    % Avoid hash index updated if hash mapping has not been changed 
    if ~(i == 1 || i == number_iterations) && sum(abs(W_last(:) - W(:))) < 1e-6
        update_table = false;
    end

    % update hash table
    if update_table
        W_last = W;
        update_iters = [update_iters, i];
        t_ = tic;

        % NOTE assuming smooth mapping
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
    end

    % KH: save intermediate model
    if ismember(i, test_iters)
        F = sprintf('%s_iter%d.mat', prefix, i);
        save(F, 'W', 'H', 'bitflips', 'train_time', 'update_time','update_iters');
        if ~opts.windows, unix(['chmod o-w ' F]); end  % matlab permission bug

        myLogInfo('[T%02d] (%d/%d) SGD %.2fs, HTU %.2fs, %d Updates #BF=%g', ...
            trialNo, i, number_iterations, train_time, update_time, numel(update_iters), bitflips);
    end

end

% KH: save final model, etc
F = [prefix '.mat'];
save(F, 'W', 'H', 'bitflips', 'train_time', 'update_time', 'test_iters','update_iters');
if ~opts.windows, unix(['chmod o-w ' F]); end % matlab permission bug
myLogInfo('[T%02d] Saved: %s\n', trialNo, F);
end
