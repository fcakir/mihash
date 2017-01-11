function [train_time, update_time, res_time, ht_updates, bits_computed_all, bitflips] = ...
    train_mutualinfo(Xtrain, Ytrain, thr_dist,  prefix, test_iters, trialNo, opts)

%%%%%%%%%%%%%%%%%%%%%%% INIT %%%%%%%%%%%%%%%%%%%%%%%
[n,d] = size(Xtrain);
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


% initialize reservoir
if reservoir_size > 0
    ind = randperm(length(train_ind));
    [reservoir, update_ind] = update_reservoir(reservoir, ...
        Xtrain(train_ind(ind(1:500)),:), Ytrain(train_ind(ind(1:500)),:), reservoir_size, W, opts.unsupervised);
    % compute new reservoir hash table (do not update yet)
end

%%%%%%%%%%%%%%%%%%%%%%% INIT %%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%% SET UP MUTUALINFO %%%%%%%%%%%%%%%%%%%%%%%
% for AdaptHash
code_length = opts.nbits;
number_iterations = opts.noTrainingPoints;
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
%%%%%%%%%%%%%%%%%%%%%%% SET UP %%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%% STREAMING BEGINS! %%%%%%%%%%%%%%%%%%%%%%%
for iter = 1:number_iterations
    
    ind = train_ind(iter);
    spoint = Xtrain(ind, :);
    if ~opts.unsupervised
        slabel = Ytrain(ind, :);
    end    
    
    % hash function update
    t_ = tic;
    input.X = spoint;
    input.Y = slabel;
   
    [output, gradient] = mutual_info(W, input, reservoir, opts.no_bins, opts.sigmf_p, ...
                                       opts.unsupervised, thr_dist, 1);
    % sgd
    W = W - opts.stepsize*gradient;
    train_time = train_time + toc(t_);


    % ---- reservoir update & compute new reservoir hash table ----
    t_ = tic;
    Hres_new = [];
    if reservoir_size > 0
        [reservoir, update_ind] = update_reservoir(reservoir, ...
            spoint, slabel, reservoir_size, W, opts.unsupervised);
        % compute new reservoir hash table (do not update yet)
        Hres_new = (reservoir.X * W > 0);
    end

    % ---- determine whether to update or not ----
    [update_table, trigger_val, h_ind] = trigger_update(iter, ...
        opts, W_lastupdate, W, reservoir, Hres_new, ...
		 opts.unsupervised, thr_dist);
    inv_h_ind = setdiff(1:opts.nbits, h_ind);  % keep these bits unchanged
    if reservoir_size > 0 && numel(h_ind) < opts.nbits  % selective update
        %assert(opts.fracHash < 1);
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




