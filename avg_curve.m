function [px, py] = avg_curve(Y, X)
	% plot the "average curve" of multiple trials
	% based on interpolating at common locations
	ntrials = size(Y, 1);
	npoints = size(Y, 2);
	if npoints == 1, px = mean(X); py = mean(Y); return, end
	px = linspace(mean(min(X,[],2)), mean(max(X,[],2)), 2*npoints);
	py = zeros(ntrials, length(px));
	for i = 1:ntrials
		% TODO this will give error if X's elements are not unique
		% trigger: X=bitflips && update_interval>test_interval
		py(i, :) = interp1(X(i, :), Y(i, :), px, 'linear', 'extrap');
	end
	py = mean(py,1);
end
