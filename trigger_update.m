function [update_table, ret_val, h_ind] = trigger_update(iter, opts, ...
    W_last, W, reservoir, Hres_new, varargin)

% Do we need to update the hash table?
% Note: The hash mapping has been updated first, so is the reservoir hash table
%
update_table = false;
ret_val = -1;
h_ind = 1:opts.nbits;

% ----------------------------------------------
% update on first iteration
if (iter == 1) %|| (iter*opts.batchSize >= opts.noTrainingPoints)
    update_table = true;  
    return;
end

% ----------------------------------------------
% no update if hash mapping has not changed
if sum(abs(W_last(:) - W(:))) < 1e-6
    update_table = false;
    return;
end

% ----------------------------------------------
% no reservoir -- use updateInterval
if opts.reservoirSize <= 0
    update_table = ~mod(iter*opts.batchSize, opts.updateInterval);
    return;
end

% ----------------------------------------------
% below: using reservoir
% the reservoir has been updated before this function
%
%if exist('bf_thr', 'var')==0 || isempty(bf_thr)
%    bf_thr = opts.flipThresh; 
%end

switch lower(opts.trigger)
    case 'bf'
        assert(~isempty(reservoir.H));
        assert(~isempty(Hres_new));

        % get bitflips
        bitdiff  = xor(reservoir.H, Hres_new);
        bitflips = sum(bitdiff(:)) / reservoir.size;

        % signal update of actual hash table, when:
        % 1) we're at an updateInterval, AND
        % 2) #bitflips > flipThresh
        if opts.updateInterval > 0 && ...
                mod(iter*opts.batchSize, opts.updateInterval) == 0
            update_table = (bitflips > opts.flipThresh);
            myLogInfo('BF=%g vs. %g, update=%d', ...
                bitflips, opts.flipThresh, update_table);
        end

        % ----------------------------------------
        % [DEPRECATED: opts.adaptive]
        % signal update of actual hash table, when:
        %
        % 1) using updateInterval ONLY (for rs_baseline)
        % 2) using updateInterval + adaptive
        % 3) #bitflips > adaptive thresh (for rs, USING adaptive threshold)
        % 4) #bitflips > flipThresh (for rs, NOT USING adaptive threshold)
        %
        % NOTE: get_opts() ensures only one scenario will happen
        %
        %if opts.updateInterval > 0 && ...
        %        mod(iter*opts.batchSize, opts.updateInterval) == 0
        %    % cases 1, 2
        %    % check whether to do an update to the hash table
        %    %
        %    if (opts.adaptive <= 0) || (opts.adaptive > 0 && bitflips > opts.flipThresh)
        %        update_table = true;
        %    end
        %elseif (opts.updateInterval <= 0) && (opts.adaptive > 0 && bitflips > opts.flipThresh)
        %    % case 3
        %    update_table = true;
        %elseif (opts.flipThresh > 0) && (bitflips > opts.flipThresh)
        %    % case 4
        %    update_table = true;
        %end
        ret_val = bitflips;

    case 'mi'
        if opts.updateInterval > 0 && ...
                mod(iter*opts.batchSize, opts.updateInterval) == 0
            [mi_impr, max_mi] = trigger_mutualinfo(iter, W, W_last, ...
                reservoir.X, reservoir.Y, reservoir.H, Hres_new, ...
                   reservoir.size, opts.nbits, varargin{:});
            update_table = mi_impr > opts.miThresh;
            myLogInfo('Max MI=%g, MI diff=%g, update=%d', max_mi, mi_impr, update_table);
            ret_val = mi_impr;
        end
    otherwise
        error(['unknown/unimplemented opts.trigger: ' opts.trigger]);
end

% regardless of trigger type, do selective hash function update
if opts.fracHash < 1
    h_ind = selective_update(reservoir.H, Hres_new, reservoir.size, ...
        opts.nbits, opts.fracHash, opts.verifyInv);
    if opts.randomHash
        h_ind = randperm(opts.nbits, length(h_ind));
    end
end
end


% -------------------------------------------------------------------------
function [mi_impr, max_mi] = trigger_mutualinfo(iter, W, W_last, X, Y, ...
    Hres, Hnew, reservoir_size, nbits, unsupervised, thr_dist)

% assertions
if exist('unsupervised', 'var') == 0 
    unsupervised = false; 
elseif unsupervised
    assert(exist('thr_dist', 'var'));
end
assert(isequal(nbits, size(Hnew,2), size(Hres,2)));
assert(isequal(reservoir_size, size(Hres,1), size(Hnew,1)));
assert((~unsupervised && ~isempty(Y)) || (unsupervised && isempty(Y)));

% take actual reservoir size into account
%reservoir_size = min(iter, reservoir_size);
%X = X(1:reservoir_size,:); Y = Y(1:reservoir_size);    
%Hres = Hres(1:reservoir_size,:); Hnew = Hnew(1:reservoir_size,:);
if ~unsupervised 
    cateTrainTrain = (repmat(Y,1,length(Y)) == repmat(Y,1,length(Y))');
else
    cateTrainTrain = squareform(pdist(X, 'euclidean')) <= thr_dist;
end
assert(isequal((W_last'*X' > 0)', Hres));

% distance
hdist = (2*Hres - 1)*(2*Hres - 1)';
hdist = (-hdist + nbits)./2;   

% if Q is the (hamming) distance - x axis
% estimate P(Q|+), P(Q|-) & P(Q)

condent = zeros(1,reservoir_size);
Qent = zeros(1, reservoir_size);
% make this faster
for j=1:reservoir_size
    A = hdist(j, :); M = A(cateTrainTrain(j, :)); NM = A(~cateTrainTrain(j, :));
    prob_Q_Cp = histcounts(M, 0:1:nbits);  % raw P(Q|+)
    prob_Q_Cn = histcounts(NM, 0:1:nbits); % raw P(Q|-)
    sum_Q_Cp  = sum(prob_Q_Cp);
    sum_Q_Cn  = sum(prob_Q_Cn);
    prob_Q    = (prob_Q_Cp + prob_Q_Cn)/(sum_Q_Cp + sum_Q_Cn);
    prob_Q_Cp = prob_Q_Cp/sum_Q_Cp;
    prob_Q_Cn = prob_Q_Cn/sum_Q_Cn;
    prob_Cp   = length(M)/(length(M) + length(NM)); %P(+)
    prob_Cn   = 1 - prob_Cp; % P(-) 

    % estimate H(Q) entropy
    idx = find(prob_Q > 0);
    Qent(j) = -sum(prob_Q(idx).*log2(prob_Q(idx)));

    % estimate H(Q|C)
    idx = find(prob_Q_Cp > 0);
    p = -sum(prob_Q_Cp(idx).*log2(prob_Q_Cp(idx)));
    idx = find(prob_Q_Cn > 0);
    n = -sum(prob_Q_Cn(idx).*log2(prob_Q_Cn(idx)));
    condent(j) = p * prob_Cp + n * prob_Cn;    
end

assert(all(Qent-condent >= 0));
assert(isequal((W'*X' > 0)', Hnew));

% estimate P(Q)
hdistn = (2*Hnew - 1)* (2*Hnew - 1)';
hdistn = (-hdistn + nbits)./2;   
condentn = zeros(1,reservoir_size);
Qentn = zeros(1, reservoir_size);
% make this faster
for j=1:reservoir_size
    A = hdistn(j, :); M = A(cateTrainTrain(j, :)); NM = A(~cateTrainTrain(j, :));
    prob_Q_Cp = histcounts(M, 0:1:nbits);  % raw P(Q|+)
    prob_Q_Cn = histcounts(NM, 0:1:nbits); % raw P(Q|-)
    sum_Q_Cp  = sum(prob_Q_Cp);
    sum_Q_Cn  = sum(prob_Q_Cn);
    prob_Q    = (prob_Q_Cp + prob_Q_Cn)/(sum_Q_Cp + sum_Q_Cn);
    prob_Q_Cp = prob_Q_Cp/sum_Q_Cp;
    prob_Q_Cn = prob_Q_Cn/sum_Q_Cn;
    prob_Cp   = length(M)/(length(M) + length(NM)); %P(+)
    prob_Cn   = 1 - prob_Cp; % P(-) 

    % estimate H(Q) entropy
    idx = find(prob_Q > 0);
    Qentn(j) = -sum(prob_Q(idx).*log2(prob_Q(idx)));

    % estimate H(Q|C)
    idx = find(prob_Q_Cp > 0);
    p = -sum(prob_Q_Cp(idx).*log2(prob_Q_Cp(idx)));
    idx = find(prob_Q_Cn > 0);
    n = -sum(prob_Q_Cn(idx).*log2(prob_Q_Cn(idx)));
    condentn(j) = p * prob_Cp + n * prob_Cn;    
end

assert(all(Qentn - condentn >= 0));
mi_impr = mean(Qentn - condentn) - mean(Qent - condent);
max_mi = mean(Qentn);

% 
%figure('Visible','off');
%bar(c_h);
%vline(find((cumsum(c_h) <= fracHash) == 0,1));
%ylim([0 1]);
%legend(sprintf('Max MI :%g, MI difference: %g, New mean MI: %g', mean(Qent), mean(Qent - condent), mean(Qentn - condentn)));
%saveas(gcf, sprintf('/research/codebooks/hashing_project/data/misc/type6-IV/hash_function_bf_%g_%05d.png', nbits, iter));
%close(gcf);
end


function h_ind = selective_update(Hres, Hnew, reservoir_size, nbits, ...
    fracHash, inverse)
% selectively update hash bits, criterion: #bitflip
% output
%   h_ind: indices of hash bits to update

% assertions
assert(ceil(nbits*fracHash) > 0);
assert(isequal(nbits, size(Hnew,2), size(Hres,2)));
assert(isequal(reservoir_size, size(Hres,1), size(Hnew,1)));

% take actual reservoir size into account
%reservoir_size = min(iter, reservoir_size);
%Hres = Hres(1:reservoir_size,:); Hnew = Hnew(1:reservoir_size,:);

% which hash functions causes the most bitflips in the reservoir
[c_h, sorted_h] = sort(sum(xor(Hnew, Hres),1),'descend');
%h_ind = sorted_h(1:ceil(fracHash*nbits));
%h_ind = 1:nbits;
c_h = c_h./ norm(c_h,1);
h_ind = sorted_h(cumsum(c_h) <= fracHash);
if isempty(h_ind), h_ind = sorted_h(1); end;
if inverse
    inv_sorted_h = fliplr(sorted_h);
    h_ind = inv_sorted_h(1:length(h_ind));
end
if ~isvector(h_ind) || any(isnan(h_ind))
    error(['Something is wrong with h_ind']);
end
end
