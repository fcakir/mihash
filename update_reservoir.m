function [reservoir, update_ind] = update_reservoir(reservoir, ...
    spoints, slabels, max_reservoir_size, W)
% 
% reservoir sampling, update step, based on random sort
% inputs:
%   reservoir: struct(X, Y, H, PQ, size)
% outputs:
%   update_ind: updated index ([] for no update)
%
assert(isstruct(reservoir));
n = size(spoints, 1);
assert(n == size(slabels, 1));

if reservoir.size < max_reservoir_size
    % if reservoir not full, append
    reservoir.X = [reservoir.X; spoints];
    reservoir.Y = [reservoir.Y; slabels];
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
            reservoir.Y(maxind, :) = slabels(i, :);
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
