function [update_table, ret_val] = trigger_update(iter, opts, ...
		W_last, W, X, Y, Hres_old, Hres_new, bf_thr)

	% Do we need to update the hash table?
	% Note: The hash mapping has been updated first, so is the reservoir hash table
	%
	update_table = false;
	ret_val = -1;

	% ----------------------------------------------
	% update on first & last iteration no matter what
	if iter == 1 || iter == opts.noTrainingPoints 
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
	if opts.reg_rs <= 0
		update_table = ~mod(iter, opts.updateInterval);
		return;
	end

	% ----------------------------------------------
	% below: using reservoir
	% the reservoir has been updated before this function
	if nargin < 9 || isempty(bf_thr)
		bf_thr = opts.flipThresh; 
	end
	
	switch lower(opts.trigger)
		case 'bf'
			assert(~isempty(Hres_old));
			assert(~isempty(Hres_new));

			% get bitflips
			bitdiff  = xor(Hres_old, Hres_new);
			bitflips = sum(bitdiff(:))/opts.reservoirSize;

			% signal update of actual hash table, when:
			%
			% 1) using updateInterval ONLY (for rs_baseline)
			% 2) using updateInterval + adaptive
			% 3) #bitflips > adaptive thresh (for rs, USING adaptive threshold)
			% 4) #bitflips > flipThresh (for rs, NOT USING adaptive threshold)
			%
			% NOTE: get_opts() ensures only one scenario will happen
			%
			if opts.updateInterval > 0 && mod(iter, opts.updateInterval) == 0
				% cases 1, 2
				% check whether to do an update to the hash table
				%
				%pret_val = ret_val;
				%ret_val = trigger_update_fatih(W, Xsample, Ysample, Hres, Hnew, reservoir_size);
				%
				if (opts.adaptive <= 0) || (opts.adaptive > 0 && bitflips > bf_thr)
					update_table = true;
				end
			elseif (opts.updateInterval <= 0) && (opts.adaptive > 0 && bitflips > bf_thr)
				% case 3
				update_table = true;
			elseif (opts.flipThresh > 0) && (bitflips > bf_thr)
				% case 4
				update_table = true;
			end
			ret_val = bitflips;
		case 'mi'
			if opts.updateInterval > 0 && mod(iter, opts.updateInterval) == 0
				[mi_impr, max_mi] = trigger_mutualinfo(W, W_last, X, Y, Hres_old, Hres_new, opts.reservoirSize, opts.nbits);
				myLogInfo('Max MI=%g, MI diff=%g', max_mi, mi_impr);
				update_table = mi_impr > opts.mi_thr;
				ret_val = mi_impr;
			end
		otherwise
			error(['unknown/unimplemented opts.trigger: ' opts.trigger]);
	end

end


function [mi_impr, max_mi] = trigger_mutualinfo(W, W_last, X, Y, Hres, Hnew, reservoir_size, nbits)
    cateTrainTrain = (repmat(Y,1,length(Y)) == repmat(Y,1,length(Y))');
    no_bits = size(Hres,2);
    assert(isequal(no_bits, size(Hnew,2)));
    assert(isequal(reservoir_size,size(Hres,1), size(Hnew,1)));
    % if Q is the (hamming) distance - x axis
    % estimate P(Q|+), P(Q|-) & P(Q)
    assert(isequal((W_last'*X' > 0)', Hres));
    hdist = (2*Hres - 1)* (2*Hres - 1)';
    hdist = (-hdist + no_bits)./2;   
    condent = zeros(1,reservoir_size);
    Qent = zeros(1, reservoir_size);
    % make this faster
    for j=1:reservoir_size
        A = hdist(j, :); M = A(cateTrainTrain(j, :)); NM = A(~cateTrainTrain(j, :));
        prob_Q_Cp = histcounts(M, 0:1:nbits, 'Normalization', 'probability'); % P(Q|+)
        prob_Q_Cn = histcounts(NM, 0:1:nbits, 'Normalization', 'probability'); % P(Q|-)
        prob_Q    = histcounts([M NM], 0:1:nbits, 'Normalization','probability'); % P(Q)        
        prob_Cp   = length(M)/(length(M) + length(NM)); %P(+)
        prob_Cn   = 1 - prob_Cp; % P(-)
        
        % estimate H(Q) entropy
        for q = 1:length(prob_Q)
            if prob_Q(q) == 0, lg = 0; else lg = log2(prob_Q(q)); end
            Qent(j) = Qent(j) - prob_Q(q) * lg;
        end
        
        % estimate H(Q|C)
        p = 0;
        for q=1:length(prob_Q_Cp)
            if prob_Q_Cp(q) == 0, lg = 0; else lg = log2(prob_Q_Cp(q)); end
            p = p - prob_Q_Cp(q) * lg;
        end
        n = 0;
        for q=1:length(prob_Q_Cn)
            if prob_Q_Cn(q) == 0, lg = 0; else lg = log2(prob_Q_Cn(q)); end
            n = n - prob_Q_Cn(q) * lg;
        end
        condent(j) = p * prob_Cp + n * prob_Cn;    
    end
    
    assert(all(Qent-condent >= 0));
    % estimate P(Q)
    assert(isequal((W'*X' > 0)', Hnew));
    hdistn = (2*Hnew - 1)* (2*Hnew - 1)';
    hdistn = (-hdistn + no_bits)./2;   
    condentn = zeros(1,reservoir_size);
    Qentn = zeros(1, reservoir_size);
    % make this faster
    for j=1:reservoir_size
        A = hdistn(j, :); M = A(cateTrainTrain(j, :)); NM = A(~cateTrainTrain(j, :));
        prob_Q_Cp = histcounts(M, 0:1:nbits, 'Normalization', 'probability'); % P(Q|+)
        prob_Q_Cn = histcounts(NM, 0:1:nbits, 'Normalization', 'probability'); % P(Q|-)
        prob_Q    = histcounts([M NM], 0:1:nbits, 'Normalization','probability'); % P(Q)        
        prob_Cp   = length(M)/(length(M) + length(NM)); %P(+)
        prob_Cn   = 1 - prob_Cp; % P(-)
        
        % estimate H(Q) entropy
        for q = 1:length(prob_Q)
            if prob_Q(q) == 0, lg = 0; else lg = log2(prob_Q(q)); end
            Qentn(j) = Qentn(j) - prob_Q(q) * lg;
        end
        
        % estimate H(Q|C)
        p = 0;
        for q=1:length(prob_Q_Cp)
            if prob_Q_Cp(q) == 0, lg = 0; else lg = log2(prob_Q_Cp(q)); end
            p = p - prob_Q_Cp(q) * lg;
        end
        n = 0;
        for q=1:length(prob_Q_Cn)
            if prob_Q_Cn(q) == 0, lg = 0; else lg = log2(prob_Q_Cn(q)); end
            n = n - prob_Q_Cn(q) * lg;
        end
        condentn(j) = p * prob_Cp + n * prob_Cn;    
    end
    
    assert(all(Qentn - condentn >= 0));
    mi_impr = mean(Qentn - condentn) - mean(Qent - condent);
    max_mi = mean(Qentn);
end
