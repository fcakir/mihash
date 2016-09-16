% -----------------------------------------------------------
% label arrival strategy
% NOTE: does not handle multi-labeled case yet
function ind = get_ordering(trialNo, Y, opts)
	labels = round(Y/10);
	labels = labels(1:opts.noTrainingPoints);
	uniqLabels = unique(labels);
	numLabels = numel(uniqLabels);

	labeledExamples = cell(1, numLabels);
	for n = 1:numLabels
		labeledExamples{n} = find(labels == uniqLabels(n));
	end

	% use the first example from the first label
	ind = 1;
	seenLabInds = 1;
	remnLabInds = 2:numLabels;
	exhausted   = cellfun(@(x) isempty(x), labeledExamples);

	% fill in from the second
	for i = 2:opts.noTrainingPoints
		% determine the next label
		if rand < opts.pObserve
			% get a new label
			L = randi([1, length(remnLabInds)]);
			newLabel = remnLabInds(L);
			assert(~ismember(newLabel, seenLabInds));
			seenLabInds = [seenLabInds, newLabel];
			remnLabInds(L) = [];
		else
			% use a seen label
			% make sure it's not an already-exhausted label
			nonempty = find(~exhausted(seenLabInds));
			assert(~isempty(nonempty), 'Seen labels are all exhausted!?');
			L = randi([1, length(nonempty)]);
			newLabel = seenLabInds(nonempty(L));
		end

		% get the next example with this label
		ind = [ind, labeledExamples{newLabel}(1)];
		labeledExamples{newLabel}(1) = [];
		exhausted(newLabel) = isempty(labeledExamples{newLabel});

		if numel(seenLabInds) == numLabels
			myLogInfo('[T%02d] All labels are seen @ t=%d/%d\n', trialNo, i, opts.noTrainingPoints);
			break;
		end
		if all(exhausted(seenLabInds))
			myLogInfo('[T%02d] Seen labels are exhausted @ t=%d/%d', trialNo, i, opts.noTrainingPoints);
			break;
		end
	end

	% second stage: randomly sample the rest
	if i < opts.noTrainingPoints
		ind = [ind, setdiff(1:opts.noTrainingPoints, ind)];
	end
	for j = numLabels:opts.noTrainingPoints
		if numel(unique(labels(ind(1:j)))) == numLabels
			myLogInfo('[T%02d] All labels are seen @ t=%d/%d\n', trialNo, j, opts.noTrainingPoints);
			break;
		end
	end
end
