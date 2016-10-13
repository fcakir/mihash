function [reservoir, update_ind] = update_reservoir(reservoir, ...
    spoints, slabels, max_reservoir_size, W, unsupervised)
% 
% reservoir sampling, update step, based on random sort
% inputs:
%   reservoir: struct(X, Y, H, PQ, size)
% outputs:
%   update_ind: updated index ([] for no update)
%
assert(isstruct(reservoir));
assert((~unsupervised && ~isempty(slabels)) || (unsupervised && isempty(slabels)));
n = size(spoints, 1);
if ~unsupervised, assert(n == size(slabels, 1)); end;

if reservoir.size < max_reservoir_size
    % if reservoir not full, append (up to max_reservoir_size)
    n = min(n, max_reservoir_size - reservoir.size);
    reservoir.X = [reservoir.X; spoints(1:n, :)];
    if ~unsupervised
        reservoir.Y = [reservoir.Y; slabels(1:n, :)];
    end
    reservoir.PQ = [reservoir.PQ; rand(n, 1)];
    update_ind = reservoir.size + (1:n);
else
    % full reservoir, update
    update_ind = [];
    for i = 1:n
        % pop max from priority queue
        [maxval, maxind] = max(reservoir.PQ);
        r = rand;
        if maxval > r
            % push into priority queue
            reservoir.PQ(maxind)   = r;
            reservoir.X(maxind, :) = spoints(i, :);
            if ~unsupervised
                 reservoir.Y(maxind, :) = slabels(i, :);
	    end
            update_ind = [update_ind, maxind];
        end
    end
end
reservoir.size = size(reservoir.X, 1);

% if hash functions are given -- udpate entries
if exist('W', 'var')
    if isempty(reservoir.H)
        reservoir.H = (reservoir.X * W > 0);
    elseif ~isempty(update_ind)
        reservoir.H(update_ind, :) = (reservoir.X(update_ind, :) * W > 0);
    end
end
