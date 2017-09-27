function [reservoir, update_ind] = update_reservoir(reservoir, ...
    points, labels, max_reservoir_size, unsupervised)
% 
% reservoir sampling, update step, based on random sort
% inputs:
%   reservoir: struct(X, Y, H, PQ, size)
% outputs:
%   update_ind: indices of updated entries ([] if no update)
%
if ~exist('unsupervised', 'var'), unsupervised = isempty(labels); end
n = size(points, 1);
if ~unsupervised, assert(n == size(labels, 1)); end;

if reservoir.size < max_reservoir_size
    % if reservoir not full, append (up to max_reservoir_size)
    n = min(n, max_reservoir_size - reservoir.size);
    reservoir.X = [reservoir.X; points(1:n, :)];
    if ~unsupervised
        reservoir.Y = [reservoir.Y; labels(1:n, :)];
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
            reservoir.X(maxind, :) = points(i, :);
            if ~unsupervised
                reservoir.Y(maxind, :) = labels(i, :);
            end
            update_ind = [update_ind, maxind];
        end
    end
end
reservoir.size = size(reservoir.X, 1);
end
