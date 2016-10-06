function train_okh(run_trial, opts)

global Xtrain Ytrain

train_time  = zeros(1, opts.ntrials);
update_time = zeros(1, opts.ntrials);
bit_flips   = zeros(1, opts.ntrials);
ht_updates  = zeros(1, opts.ntrials);
bit_recomp  = zeros(1, opts.ntrials);
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

    % do optimization
    [train_time(t), update_time(t), ht_updates(t), bit_recomp(t), bit_flips(t)] ...
        = OKH(Xtrain, Ytrain, prefix, test_iters, t, opts);
end

myLogInfo('Training time (total): %.2f +/- %.2f', mean(train_time), std(train_time));
myLogInfo('HTupdate time (total): %.2f +/- %.2f', mean(update_time), std(update_time));
if strcmp(opts.mapping, 'smooth')
    myLogInfo('    Hash Table Updates (per): %.4g +/- %.4g', mean(ht_updates), std(ht_updates));
    myLogInfo('    Bit Recomputations (per): %.4g +/- %.4g', mean(bit_recomp), std(bit_recomp));
    myLogInfo('    Bit flips (per): %.4g +/- %.4g', mean(bit_flips), std(bit_flips));
end
end


% ---------------------------------------------------------
% ---------------------------------------------------------
function [train_time, update_time, ht_updates, bits_computed_all, bitflips] = ...
    OKH(Xtrain, Ytrain, prefix, test_iters, trialNo, opts)

%%%%%%%%%%%%%%%%%%%%%%% GENERIC INIT %%%%%%%%%%%%%%%%%%%%%%%
% are we handling a mult-labeled dataset?
multi_labeled = (size(Ytrain, 2) > 1);
if multi_labeled, myLogInfo('Handling multi-labeled dataset'); end

% set up reservoir
reservoir = [];
reservoir_size = opts.reservoirSize;
if reservoir_size > 0
    reservoir.size = 0;
    reservoir.X    = [];
    reservoir.Y    = [];
    reservoir.PQ   = [];
    reservoir.H    = [];  % mapped binary codes for the reservoir
end

% order training examples
if opts.pObserve > 0
    % [OPTIONAL] order training points according to label arrival strategy
    train_ind = get_ordering(trialNo, Ytrain, opts);
else
    % randomly shuffle training points before taking first noTrainingPoints
    % this fixes issue #25
    train_ind = randperm(size(Xtrain, 1), opts.noTrainingPoints);
end
%%%%%%%%%%%%%%%%%%%%%%% GENERIC INIT %%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%% SET UP OKH %%%%%%%%%%%%%%%%%%%%%%%
tic;
% do kernel mapping to Xtrain
% KX: each COLUMN is a kernel-mapped training example
[KX, Xanchor] = init_okh(Xtrain);
para.c        = opts.c; %0.1;
para.alpha    = opts.alpha; %0.2;
para.anchor   = Xanchor;

% for recording time
update_time = 0;
train_time  = toc;  
myLogInfo('Preprocessing took %f sec', train_time);

number_iterations = opts.noTrainingPoints/2;
myLogInfo('[T%02d] %d training iterations', trialNo, number_iterations);

d = size(KX, 1);
W = rand(d, opts.nbits)-0.5;
% NOTE: W_lastupdate keeps track of the last W used to update the hash table
%       W_lastupdate is NOT the W from last iteration
W_lastupdate = W;
stepW = zeros(size(W));  % Gradient accumulation matrix
H = [];

% bit flips & bits computed
bitflips          = 0;
bitflips_res      = 0;
bits_computed_all = 0;

% HT updates
update_iters = [];
h_ind_array  = [];
%%%%%%%%%%%%%%%%%%%%%%% SET UP OKH %%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%% STREAMING BEGINS! %%%%%%%%%%%%%%%%%%%%%%%
%rX = KX(:,idxTrain); %set being search in testing 
%tX = KX(:,idxTest); %query set in testing
for i = 1:number_iterations
    idx_i = Ytrain(2*i-1, :); %idxTrain(dataIdx(2*i-1));
    idx_j = Ytrain(2*i, :);   %idxTrain(dataIdx(2*i));
    s = 2*(idx_i==idx_j)-1;

    xi = KX(:, 2*i-1); %KX(:,idx_i);
    xj = KX(:, 2*i);   %X(:,idx_j);

    % hash function update
    t_ = tic;
    W = OKHlearn(xi,xj,s,W,para);
    train_time = train_time + toc(t_);


    % ---- reservoir update & compute new reservoir hash table ----
    if reservoir_size > 0
        [reservoir, update_ind] = update_reservoir(reservoir, [xi,xj]', ...
            [idx_i; idx_j], reservoir_size, W_lastupdate);
        % compute new reservoir hash table (do not update yet)
        Hres_new = (W' * reservoir.X' > 0)';
    end

    % ---- determine whether to update or not ----
    [update_table, trigger_val, h_ind] = trigger_update(i, opts, ...
        W_lastupdate, W, reservoir, Hres_new);
    inv_h_ind = setdiff(1:opts.nbits, h_ind);  % keep these bits unchanged
    if reservoir_size > 0 && numel(h_ind) < opts.nbits  % selective update
        assert(opts.fracHash < 1);
        Hres_new(:, inv_h_ind) = reservoir.H(:, inv_h_ind);
    end

    % ---- hash table update, etc ----
    if update_table
	h_ind_array = [h_ind_array; single(ismember(1:opts.nbits, h_ind))];

        % update W
        W_lastupdate(:, h_ind) = W(:, h_ind);
        if opts.accuHash > 0 && ~isempty(inv_h_ind) % gradient accumulation
            assert(sum(sum(abs((W_lastupdate - stepW) - W))) < 1e-10);
            stepW(:, inv_h_ind) = W_lastupdate(:, inv_h_ind) - W(:, inv_h_ind);
        end
        W = W_lastupdate;
        update_iters = [update_iters, i];

        % update reservoir hash table
        if reservoir_size > 0
            reservoir.H = Hres_new;
            if strcmpi(opts.trigger, 'bf')
                bitflips_res = bitflips_res + trigger_val;
            end
        end

        % actual hash table update (record time)
        t_ = tic;
        [H, bf_all, bits_computed] = update_hash_table(H, W, KX', Ytrain, ...
            h_ind, update_iters, opts);
        bits_computed_all = bits_computed_all + bits_computed;
        bitflips = bitflips + bf_all;
        update_time = update_time + toc(t_);
    end

    % ---- save intermediate model ----
    if ismember(i, test_iters)
        F = sprintf('%s_iter%d.mat', prefix, i);
        save(F, 'W', 'H', 'bitflips', 'bits_computed_all', ...
            'train_time', 'update_time', 'update_iters');
        % fix permission
        if ~opts.windows, unix(['chmod g+w ' F]); unix(['chmod o-w ' F]); end

        myLogInfo('[T%02d] (%d/%d) OKH %.2fs, HTU %.2fs, %d Updates #BF=%g', ...
            trialNo, i, number_iterations, train_time, update_time, numel(update_iters), bitflips);
    end
end
%%%%%%%%%%%%%%%%%%%%%%% STREAMING ENDED! %%%%%%%%%%%%%%%%%%%%%%%

% save final model, etc
F = [prefix '.mat'];
save(F, 'W', 'H', 'bitflips', 'bits_computed_all', ...
    'train_time', 'update_time', 'test_iters', 'update_iters', ...
    'h_ind_array');
% fix permission
if ~opts.windows, unix(['chmod g+w ' F]); unix(['chmod o-w ' F]); end

ht_updates = numel(update_iters);
myLogInfo('%d Hash Table updates, bits computed: %g', ht_updates, bits_computed_all);
myLogInfo('[T%02d] Saved: %s\n', trialNo, F);
end


% ---------------------------------------------------------
% ---------------------------------------------------------
function [KX, Xanchor] = init_okh(Xtrain)
assert(size(Xtrain, 1) >= 4000);

tic;
% sample support samples (300) from the FIRST HALF of training set
nhalf = floor(size(Xtrain, 1)/2);
ind = randperm(nhalf, 300);
Xanchor = Xtrain(ind, :);

% estimate sigma for Gaussian kernel using samples from the SECOND HALF
ind = randperm(nhalf, 2000);
Xval = Xtrain(nhalf+ind, :);
Kval = sqdist(Xval', Xanchor');
sigma = mean(mean(Kval, 2));
myLogInfo('Estimated sigma = %g', sigma);
clear Xval Kval

% preliminary for testing
% kernel mapping the whole set
KX = exp(-0.5*sqdist(Xtrain', Xanchor')/sigma^2)';
KX = [KX; ones(1,size(KX,2))];
end

