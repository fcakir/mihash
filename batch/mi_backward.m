function [bot] = mi_backward(layer, bot, top)

X = squeeze(bot.x);  % nbitsxN
[nbits, N] = size(X);

opts = layer.opts;
onGPU = numel(opts.gpus) > 0;

no_bins = opts.nbins;
deltaD = nbits / no_bins;
centersD = 0: deltaD: nbits;

pD   = top.aux.pD;
pDCp = top.aux.pDCp;
pDCn = top.aux.pDCn;
prCp = top.aux.prCp;
prCn = top.aux.prCn;
phi  = top.aux.phi;
Xp   = top.aux.Xp;
Xn   = top.aux.Xn;
hdist = top.aux.distance;

minus1s = -ones(size(pD));
if onGPU, minus1s = gpuArray(minus1s); end

% 1. H/P(D)
d_H_pD = deriv_ent(pD, minus1s);  % NxL

% 2. H/P(D|+), H/P(D|-)
d_H_pDCp = diag(prCp) * d_H_pD;
d_H_pDCn = diag(prCn) * d_H_pD;

% 3. Hcond/P(D|+), Hcond/P(D|-)
d_Hcond_pDCp = diag(prCp) * deriv_ent(pDCp, minus1s);
d_Hcond_pDCn = diag(prCn) * deriv_ent(pDCn, minus1s);

% 4. -MI/P(D|+), -MI/P(D|-): LxN
d_L_pDCp = -(d_H_pDCp - d_Hcond_pDCp);
d_L_pDCn = -(d_H_pDCn - d_Hcond_pDCn);

% 4.1 max_diff term
if opts.maxdif > 0
    d_Z_pDCp = zeros(N, no_bins+1);
    d_Z_pDCn = zeros(N, no_bins+1);
    if onGPU
        d_Z_pDCp = gpuArray(d_Z_pDCp);
        d_Z_pDCn = gpuArray(d_Z_pDCn);
    end
    pp = top.aux.pp;
    pn = top.aux.pn;
    Zs = -nbits: deltaD: nbits;
    for l = 0:no_bins
        % d_p(D|+,l) = \sum_{k=-L}^L z(k)*P(D|-,l+k)
        % d_p(D|-,l) = \sum_{k=-L}^L z(k)*P(D|+,l-k)
        % where OOB entries are 0
        % Note: reuse pp, pn from forward pass
        pn_l = circshift(pn, -l, 2);  % Nx(2L+1)
        pp_l = circshift(pp, -l, 2);  % Nx(2L+1)
        d_Z_pDCp(:, l+1) = pn_l * Zs';  % Nx1
        d_Z_pDCn(:, l+1) = pp_l * fliplr(Zs)';  % Nx1
    end
    d_L_pDCp = d_L_pDCp - opts.maxdif * d_Z_pDCp/nbits;
    d_L_pDCn = d_L_pDCn - opts.maxdif * d_Z_pDCn/nbits;
end

% 4.2 min_plus term
if opts.minplus > 0
    g = repmat(centersD, [N 1]);
    d_L_pDCp = d_L_pDCp + opts.minplus * g/nbits;
end

% 5. precompute dTPulse tensor
d_L_phi = zeros(nbits, N);
if onGPU, d_L_phi = gpuArray(d_L_phi); end
Np = sum(Xp, 2);
Nn = sum(Xn, 2);
invalid = (Np==0) | (Nn==0);
for l = 1:no_bins+1
    % NxN matrix of delta_hat(i, j, l) for fixed l
    dpulse = dTriPulse(hdist, centersD(l), deltaD);  % NxN
    ddp = dpulse .* Xp;  % delta_l^+(i, j)
    ddn = dpulse .* Xn;  % delta_l^-(i, j)

    alpha_p = d_L_pDCp(:, l)./Np;  alpha_p(invalid) = 0;
    alpha_n = d_L_pDCn(:, l)./Nn;  alpha_n(invalid) = 0;
    alpha_p = diag(alpha_p);
    alpha_n = diag(alpha_n);
    Ap = ddp * alpha_p + alpha_p * ddp;  % 1st term: j=i, 2nd term: j~=i
    An = ddn * alpha_n + alpha_n * ddn;

    % accumulate gradient
    d_L_phi = d_L_phi - 0.5 * phi * (Ap + An);
end

% 6. -MI/x
sigmoid = (phi+1)/2;
d_phi_x = 2 .* sigmoid .* (1-sigmoid) * opts.sigmf_p(1);  % nbitsxN
d_L_x   = d_L_phi .* d_phi_x;

% 7. final
bot.dzdx = zeros(size(bot.x), 'single');
if onGPU, bot.dzdx = gpuArray(bot.dzdx); end
bot.dzdx(1, 1, :, :) = single(d_L_x);
end


function y = dTriPulse(D, mid, delta);
% vectorized version
% mid: scalar bin center
%   D: can be a matrix
ind1 = (D > mid-delta) & (D <= mid);
ind2 = (D > mid) & (D <= mid+delta);
y = (ind1 - ind2) / delta;
end


function dHp = deriv_ent(p, minus1s)
% derivative of entropy
dHp = minus1s;
dHp(p>0) = dHp(p>0) - log2(p(p>0));
end
