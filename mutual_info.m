function [output, gradient] = mutual_info(W_last, input, reservoir, no_bins, sigmf_p,...
                                       unsupervised, thr_dist, bool_gradient)
% W             : matrix, contains hash function parameters
% input.X       : data point
% input.Y       : label of X, empty if X has no labels
% reservoir.X   : reservoir data points, reservoir_size x dimensionality
% reservoir.Y   : reservoir labels corresponding to X, if empty then X does
%                 not have labels 
% reservoir.H   : reservoir hash table, reservoir_size x nbits
% reservoir.size: number of items in the reservoir
% sigmf_p 	: two length numeric vector, e.g., [a c], to be used in sigmf
% unsupervised  : 1 if neighborhood is defined using thr_dist see below
% thr_dist      : numeric value for thresholding


% compute loss
% compute distances

output = []; gradient = [];
% assertions
if exist('unsupervised', 'var') == 0 
    unsupervised = false; 
elseif unsupervised
    assert(exist('thr_dist', 'var') == 1);
end


X = input.X;
Y = input.Y;

%assert(isequal(nbits, size(Hnew,2), size(Hres,2)));
%assert(isequal(reservoir_size, size(Hres,1), size(Hnew,1)));
%assert((~unsupervised && ~isempty(Y)) || (unsupervised && isempty(Y)));

% take actual reservoir size into account
%reservoir_size = min(iter, reservoir_size);
%X = X(1:reservoir_size,:); Y = Y(1:reservoir_size);    
%Hres = Hres(1:reservoir_size,:); Hnew = Hnew(1:reservoir_size,:);

if ~unsupervised     
    % if multilabel
    if size(reservoir.Y,2) > 1
        catePointTrain = (reservoir.Y' * Y > 0);    
    % if multiclass
    else
        catePointTrain = (reservoir.Y == Y);
    end
else
    catePointTrain = squareform(pdist2(X, reservoir.X, 'euclidean')) <= thr_dist;
end

%assert(isequal((W_last'*reservoir.X' > 0)', reservoir.H));
%Hres = reservoir.H;
% RELAXED hash codes to interval [-1, 1]
Hres = sigmf(W_last'*reservoir.X', sigmf_p)';
nbits = size(Hres, 2);

% RELAXED hash input point X
%hashX = (W_last'*X' > 0)';
hashX = sigmf(W_last'*X', sigmf_p)'; % row vector, 1 x nbits

% compute distances from hash codes
hdist = (2*hashX - 1)*(2*Hres - 1)'; % row vector, 1 x reservoir_size
hdist = (-hdist + nbits)./2;   
assert(all(hdist >= 0) && all(hdist <= nbits));
% if Q is the (hamming) distance - x axis
% estimate P(Q|+), P(Q|-) & P(Q)

% manually
pQ = zeros(1, no_bins+1); 
deltaQ = nbits / no_bins;
bordersQ = 0:deltaQ:nbits;
assert(length(bordersQ) == no_bins+1);
for i=1:no_bins+1
    pQ(i) = sum(triPulse(bordersQ(i) - deltaQ, bordersQ(i) + deltaQ, hdist));
    if pQ(i)< 0
        keyboard
    end
end
pQ = pQ ./ sum(pQ);
%if any(pQ < 0)
%    pQ = pQ - min(pQ);
%    pQ = pQ ./ sum(pQ);
%end
M = hdist(catePointTrain); NM = hdist(~catePointTrain);
pQCp = zeros(1, no_bins+1);
pQCn = zeros(1, no_bins+1);
for i=1:no_bins+1
    pQCp(i) = sum(triPulse(bordersQ(i) - deltaQ, bordersQ(i) + deltaQ, M));
    pQCn(i) = sum(triPulse(bordersQ(i) - deltaQ, bordersQ(i) + deltaQ, NM));
end

prCp = length(M) ./ (length(M) + length(NM));
prCn = 1 - prCp;
pQCp = pQCp ./ sum(pQCp);
pQCn = pQCn ./ sum(pQCn);
%if any(pQCp < 0)
%    pQCp = pQCp - min(pQCp);
%    pQCp = pQCp ./ sum(pQCp);
%end
%if any(pQCn < 0)
%    pQCn = pQCn - min(pQCn);
%    pQCn = pQCn ./ sum(pQCn);
%end
assert(sum(abs((pQCp*prCp+ pQCn*prCn) - pQ)) < 1e-6);
%if (sum(abs((pQCp*prCp+ pQCn*prCn) - pQ)) > 1e-6)
%    pQ = pQCp*prCp + pQCn*prCn;
%end

% estimate H(Q) entropy -> Qent
idx = find(pQ > 0);
Qent = -sum(pQ(idx).*log2(pQ(idx)));

% estimate H(Q|C) -> condent
idx = find(pQCp > 0);
p = -sum(pQCp(idx).*log2(pQCp(idx)));
idx = find(pQCn > 0);
n = -sum(pQCn(idx).*log2(pQCn(idx)));
condent = p * prCp + n * prCn;    

assert(Qent-condent >= 0);
output = Qent - condent;

Hres = Hres'; % nbits x reservoir_size
% Assumes hash codes are relaxed from {-1, 1} to [-1, 1]
if bool_gradient
	d_dh_phi = -0.5*Hres;
	d_delta_phi = zeros(no_bins+1, nbits, reservoir.size);
	d_pQCp_phi = zeros(no_bins+1, nbits);
	d_pQCn_phi = zeros(no_bins+1, nbits);
	for i=1:no_bins+1
		d_delta_phi(i,:,:) = (diag(dTPulse(bordersQ(i) - deltaQ, bordersQ(i) + deltaQ, hdist)) * d_dh_phi')';
	end

	for i=1:no_bins+1
		A = squeeze(d_delta_phi(i,:,:));
		d_pQCp_phi(i,:) = sum(A(:, catePointTrain),2)'./length(M);%row vector
		d_pQCn_phi(i,:) = sum(A(:, ~catePointTrain),2)'./length(NM); %row vector        
	end

	d_pQ_phi = d_pQCp_phi*prCp + d_pQCn_phi*prCn;
	t_log = zeros(1, no_bins+1);
	idx = find(pQ > 0);
	t_log(idx) = log2(pQ(idx));
	t_log = t_log+1;
	d_H_phi = sum(diag(t_log) * d_pQ_phi, 1)'; % row vector

	t_log_p = zeros(1, no_bins+1);
	t_log_n = zeros(1, no_bins+1);
	idx = find(pQCp > 0);
	idx2 = find(pQCn > 0);
	t_log_p(idx) = log2(pQCp(idx));
	t_log_p = t_log_p + 1;
	t_log_n(idx2) = log2(pQCn(idx2));
	t_log_n = t_log_n + 1;

	d_cond_phi = prCp * sum(diag(t_log_p) * d_pQCp_phi,1)' + ...
		prCn* sum(diag(t_log_n)*d_pQCn_phi, 1)';

	d_MI_phi = d_H_phi - d_cond_phi;
	ty = sigmf_p(1) * (W_last'*X' - sigmf_p(2)); % a vector
	gradient = (diag(d_MI_phi) * (diag(sigmf(ty, [1 0]) .* sigmf(ty, [1 0]) .* sigmf_p(1)) ...
			* repmat(X', 1, length(ty))'))'; % a gradient matrix
end


%ind_l = ceil(hdist./deltaQ);
%ind_l(ind_l == 0) = 1;
%ind_u = ind_l + 1;
%pQ(ind_l) = mod(hdist, deltaQ)./deltaQ;
%pQ(ind_u) = 1 - mod(hdist, deltaQ)./deltaQ;

% matlab

%condent = 0;
%Qent = 0;
% make this faster

% prob_Q_Cp = histcounts(M, 0:1:nbits);  % raw P(Q|+)
% prob_Q_Cn = histcounts(NM, 0:1:nbits); % raw P(Q|-)
% sum_Q_Cp  = sum(prob_Q_Cp);
% sum_Q_Cn  = sum(prob_Q_Cn);
% prob_Q    = (prob_Q_Cp + prob_Q_Cn)/(sum_Q_Cp + sum_Q_Cn);
% prob_Q_Cp = prob_Q_Cp/sum_Q_Cp;
% prob_Q_Cn = prob_Q_Cn/sum_Q_Cn;
% prob_Cp   = length(M)/(length(M) + length(NM)); %P(+)
% prob_Cn   = 1 - prob_Cp; % P(-)
% keyboard
end


function y = triPulse(a,c,x)
% a must be smaller than c
	mid = (a+c)/2;
	%der = 2/(c-a);
	y = zeros(1, length(x));
    ind = single(x > a) .* single(x <= mid) == 1;    
	y(ind) = 1-abs(x(ind)-mid)./(c-mid);
    
    ind = single(x > mid) .* single(x <= c) == 1;
	y(ind) = 1-abs(x(ind)-mid)./(c-mid);
    if any(y < 0)
        disp('check triPulse');
        keyboard;
    end
end

function y = dTPulse(a,c,x)
% a must be smaller than c
	mid = (a+c)/2;
	der = 2/(c-a);
	y = zeros(1, length(x));
	y(single(x > a) .* single(x <= mid) == 1) = der;
	y(single(x > mid) .* single(x <= c) == 1) = -der;
end

