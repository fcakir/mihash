function [px, py] = avg_curve(Y, X)
	% plot the "average curve" of multiple trials
	% based on interpolating at common locations
	ntrials = size(Y, 1);
	npoints = size(Y, 2);
	if npoints == 1, px = mean(X); py = mean(Y); return, end
	px = linspace(mean(min(X,[],2)), mean(max(X,[],2)), 2*npoints);
	py = zeros(ntrials, length(px));
	for i = 1:ntrials
		% fix for identical values in X: artificially add small increment
		deltas = X(i, 2:end) - X(i, 1:end-1);
		for j = find(deltas <= 0)
			X(i, j+1) = X(i, j) + mean(deltas)/10;
		end
		py(i, :) = interp1(X(i, :), Y(i, :), px, 'linear', 'extrap');
	end
	py = mean(py,1);
end
