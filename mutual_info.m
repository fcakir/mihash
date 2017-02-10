function [output, gradient] = mutual_info(W_last, input, reservoir, no_bins, sigmf_p,...
                                       unsupervised, thr_dist, max_dif, bool_gradient)
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
% max_dif       : maximize the separability between the conditionals
% bool_gradient : return gradient


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
        catePointTrain = (reservoir.Y * Y' > 0);    
    % if multiclass
    else
        catePointTrain = (reservoir.Y == Y);
    end
else
    catePointTrain = pdist2(X, reservoir.X, 'euclidean') <= thr_dist;
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
if sum(pQCp) ~= 0, pQCp = pQCp ./ sum(pQCp); end;
if sum(pQCn) ~= 0, pQCn = pQCn ./ sum(pQCn); end;
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
output = -(Qent - condent); % we're minimizing the mutual info.

% generate figs
%f = figure('visible','off');

%close f;

% ToDO consider max_dif in calculating the objective value
if max_dif
	pZ = zeros(1, no_bins+1);

	for z=1:no_bins
		pZCn1 = circshift(pQCn,-z, 2);
		pZCn1(end-z+1:end) = 0;
		pZCn2 = circshift(pQCn, z, 2);
		pZCn2(1:z) = 0;
		pZ(z+1) = sum(pZCn1 .* pQCp) + sum(pZCn2 .* pQCp);
    end
    pZ(1) = sum(pQCn .* pQCp);
	output = output - max_dif * sum(bsxfun(@times, pZ, 0:1:no_bins));
end


Hres = Hres'; % nbits x reservoir_size
% Assumes hash codes are relaxed from {-1, 1} to [-1, 1]
if bool_gradient
	d_dh_phi = -0.5*Hres; % nbits x reservoir_size matrix: each column is --> \partial d_h(x, x^r) / \partial \Phi(x) = -\Phi(x^r) / 2
	d_delta_phi = zeros(nbits, reservoir.size, no_bins+1);
	d_pQCp_phi = zeros(no_bins+1, nbits);
	d_pQCn_phi = zeros(no_bins+1, nbits);
	for i=1:no_bins+1
        A = dTPulse(bordersQ(i) - deltaQ, bordersQ(i) + deltaQ, hdist);
        % each column of below matrix (RHS) --> [\partial \delta_{x^r,l} / \partial d_h(x, x^r)] x [\partial d_h(x, x^r) / \partial \Phi(x)] 
        % = \partial \delta_{x^r,l} / \partial \Phi(x)
        d_delta_phi(:,:,i) = bsxfun(@times, d_dh_phi, A); 
	end

	for i=1:no_bins+1
        % Eq. 9 in report: \partial p_{D,l}^+ / \partial \Phi(x)
        % having computed d_delta_phi, we just some the respective columns
        % that correspond to positive neighbors. 
        if length(M) ~= 0, d_pQCp_phi(i,:) = sum(d_delta_phi(:, catePointTrain, i),2)'./length(M); end;%row vector
        % similar to above computation but for \partial p_{D,l}^- /
        % \partial Phi(x)
		if length(NM) ~= 0, d_pQCn_phi(i,:) = sum(d_delta_phi(:, ~catePointTrain, i),2)'./length(NM); end; %row vector        
    end
    % Eq. 8 \partial p_{D,l} / \partial \Phi(x), computed from Eq. 9
	d_pQ_phi = d_pQCp_phi*prCp + d_pQCn_phi*prCn;
	t_log = ones(1, no_bins+1);
	idx = find(pQ > 0);
	t_log(idx) = t_log(idx) + log2(pQ(idx));
    
    % \partial H(D) / \Phi(x), see Eq. 6 and 7
	d_H_phi = sum(bsxfun(@times, d_pQ_phi, t_log'), 1)'; % row vector, this is equal to negative gradient of entropy -grad H

	t_log_p = ones(1, no_bins+1);
	t_log_n = ones(1, no_bins+1);
	idx = find(pQCp > 0);
	idx2 = find(pQCn > 0);
	t_log_p(idx) = t_log_p(idx) + log2(pQCp(idx));
	t_log_n(idx2) = t_log_n(idx2) + log2(pQCn(idx2));
    
    % \partial H(D|C) / \partial \Phi(x)
	d_cond_phi = prCp * sum(bsxfun(@times, d_pQCp_phi, t_log_p'),1)' + ...
		prCn* sum(bsxfun(@times, d_pQCn_phi, t_log_n'), 1)'; % This is equal to negative gradient of cond entropy -grad H(|)

    % Eq. 6 - a vector
	d_MI_phi = d_H_phi - d_cond_phi; % This is equal to the gradient of negative MI, 
	
	% calculate gradient to maximize expected distance between the conditionals pQCp and pQCn
	if max_dif 
		d_pZ_phi = zeros(no_bins+1, nbits); 
		for z = 1:no_bins % original cross correlation requires -inf to +inf, but this gives the same result
			pZCn1 = circshift(pQCn,-z, 2); % P(d+z|-)
			pZCn1(end-z+1:end) = 0;
			pZCn2 = circshift(pQCn, z, 2); % P(d-z|-)
			pZCn2(1:z) = 0;
			
			d_ZCn_phi1 = circshift(d_pQCn_phi, -z, 1); % \delta P(d+z|-)/ \delta \Phi
			d_ZCn_phi1(:, end-z+1:end) = 0;
			d_ZCn_phi2 = circshift(d_pQCn_phi, z, 1); % \delta P(d-z|-)/ \delta \Phi
			d_ZCn_phi2(:, 1:z) = 0;

			d_pZ_phi(z+1, :) = 0*sum(bsxfun(@times, d_ZCn_phi1', pQCp)' + bsxfun(@times, d_pQCp_phi', pZCn1)', 1)' + ...
						sum(bsxfun(@times, d_ZCn_phi2', pQCp)' + bsxfun(@times, d_pQCp_phi', pZCn2)', 1)'; 
        end
        d_pZ_phi(1,:) = sum(bsxfun(@times, d_pQCn_phi', pQCp)' + bsxfun(@times, d_pQCp_phi', pQCn)',1)';
		d_MI_phi = d_MI_phi - max_dif * sum(bsxfun(@times, d_pZ_phi', 0:1:no_bins), 2);
        
	end
	% Since \Phi(x) = [\phi_1(x),...,\phi_b(x)] where \phi_i(x) = \sigma(w_i^t \times x)
    % take gradient of each \phi_i wrt to weight w_i, and multiply the
    % resulting vector with corresponding entry in d_MI_phi
	ty = sigmf_p(1) * (W_last'*X' - sigmf_p(2)); % a vector
	gradient = (bsxfun(@times, bsxfun(@times, repmat(X', 1, length(ty)), ...
        (sigmf(ty, [1 0]) .* (1 - sigmf(ty, [1 0])) .* sigmf_p(1))'), d_MI_phi'));
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


function y = triPulseV(a,c,x)
    mid = (a+c) ./ 2;
    y = zeros(length(mid),length(x));
    ind = bsxfun(@gt, x, a) & bsxfun(@le, x, mid);
    %y(:, ind) = 1-abs
end

function y = triPulse(a,c,x)
% a must be smaller than c
	mid = (a+c)/2;
	%der = 2/(c-a);
	y = zeros(1, length(x));
    ind = x > a & x <= mid;    
	y(ind) = 1-abs(x(ind)-mid)./(c-mid);
    
    ind = x > mid & x <= c;
	y(ind) = 1-abs(x(ind)-mid)./(c-mid);
    %if any(y < 0)
    %    disp('check triPulse');
    %    keyboard;
    %end
end

function y = dTPulse(a,c,x)
% a must be smaller than c
	mid = (a+c)/2;
	der = 2/(c-a);
	y = zeros(1, length(x));
	y(x > a & x <= mid) = der;
	y(x > mid & x <= c) = -der;
end

