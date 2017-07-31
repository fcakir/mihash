function [precision, recall, ap] = CalcPrecallAP(score, truth, pos)

% number of true samples
num_truesamples = length(truth);

% score is the computed hamming distance
[~, sorted_ind] = sort(score);

%sorted_truefalse=ismember(sorted_ind, truth);
truth_ext = zeros(size(sorted_ind));
truth_ext(truth) = 1;
sorted_truefalse = truth_ext(sorted_ind);

truepositive = cumsum(sorted_truefalse);

recall = truepositive(pos) / num_truesamples;
precision = truepositive(pos) ./ (pos);%[1:numds];

pos = find(sorted_truefalse>0);
ap = mean(truepositive(pos)./(pos));

end