function update_table = trigger_update(iter, opts, ...
		W_last, W, Hres_old, Hres_new)

	% Do we need to update the hash table?
	% Note: The hash mapping has been updated first, so is the reservoir hash table
	%
	update_table = false;

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
	assert(~isempty(Hres_old));
	assert(~isempty(Hres_new));

	% get bitflips
	bitdiff = xor(Hres_old, Hres_new);
	bf_temp = sum(bitdiff(:))/reservoir_size;


	% (Kun: what is this doing?)
	%THR = table_thr(max(1, length(seenLabels)));
	

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
		if (opts.adaptive <= 0) || ...
				(opts.adaptive > 0 && bf_temp > table_thr(max(1, length(seenLabels))))
			update_table = true;
			res_bf = bf_temp;
		end
	elseif (opts.updateInterval <= 0) && ...
			(opts.adaptive > 0 && bf_temp > table_thr(max(1, length(seenLabels))))
		% case 3
		update_table = true;
		res_bf = bf_temp;
	elseif (opts.flipThresh > 0) && (bf_temp > opts.flipThresh)
		% case 4
		update_table = true;
		res_bf = bf_temp;
	end

end
