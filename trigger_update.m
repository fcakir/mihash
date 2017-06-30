function [update_table, ret_val, h_ind] = trigger_update(iter, opts, ...
    W_last, W, reservoir, Hres_new, varargin)

% Do we need to update the hash table?
% Note: The hash mapping has been updated first, so is the reservoir hash table
%
update_table = false;
ret_val = -1;
nbits = opts.nbits*opts.no_blocks;
h_ind = 1:nbits;

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
        ret_val = bitflips;

    otherwise
        error(['unknown/unimplemented opts.trigger: ' opts.trigger]);
end
end
