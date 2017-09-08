function update_table = trigger_update(iter, W_last, W, reservoir, ...
    Hres_new, opts)

update_table = false;

% ----------------------------------------------
% no update if hash mapping has not changed
if sum(abs(W_last(:) - W(:))) < 1e-6
    update_table = false;
    logInfo('[W no change] iter %d, update = 0', iter);
    return;
end

% ----------------------------------------------
% at an update interval

if opts.reservoirSize <= 0 || strcmp(opts.trigger, 'fix')
    % no reservoir or 'fix' -- update
    update_table = true;
    logInfo('[Fix] iter %d, update = 1', iter);

elseif opts.reservoirSize > 0 && opts.updateInterval > 0
    % using reservoir + MI criterion
    % signal update of hash table, if MI improvement > threshold
    assert(strcmp(opts.trigger, 'mi'));

    % affinity matrix
    Aff = affinity(reservoir.X, reservoir.X, reservoir.Y, reservoir.Y, opts);

    % MI improvement
    mi_old  = eval_mutualinfo(reservoir.H, Aff);
    mi_new  = eval_mutualinfo(Hres_new, Aff);
    mi_impr = mi_new - mi_old;

    % update?
    update_table = mi_impr > opts.miThresh;
    logInfo('[MI] iter %d, improvement = %g, update = %d', iter, mi_impr, update_table);
end

end


% --------------------------------------------------------------------
function mi = eval_mutualinfo(H, affinity)
% distance
[num, nbits] = size(H);
hdist = (2*H - 1) * (2*H - 1)';
hdist = (nbits - hdist) / 2;   

% let Q be the Hamming distance
% estimate P(Q|+), P(Q|-) & P(Q)
condent = zeros(1, num);
Qent = zeros(1, num);
for j = 1:num
    D  = hdist(j, :); 
    M  = D( affinity(j, :)); 
    NM = D(~affinity(j, :));
    prob_Q_Cp = histcounts(M,  0:1:nbits);  % raw P(Q|+)
    prob_Q_Cn = histcounts(NM, 0:1:nbits);  % raw P(Q|-)
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
    p   = -sum(prob_Q_Cp(idx).*log2(prob_Q_Cp(idx)));
    idx = find(prob_Q_Cn > 0);
    n   = -sum(prob_Q_Cn(idx).*log2(prob_Q_Cn(idx)));
    condent(j) = p * prob_Cp + n * prob_Cn;    
end

mi = Qent - condent;
mi(mi < 0) = 0;  % deal with numerical inaccuracies
mi = mean(mi);
end
