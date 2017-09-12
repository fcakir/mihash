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

% 5. precompute dTPulse tensor
d_L_phi = zeros(nbits, N);
if onGPU, d_L_phi = gpuArray(d_L_phi); end
Np = sum(Xp, 2);
Nn = sum(Xn, 2);
invalid = (Np==0) | (Nn==0);
for l = 1:no_bins+1
    % NxN matrix of delta_hat(i, j, l) for fixed l
    dpulse = triPulseDeriv(hdist, centersD(l), deltaD);  % NxN
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
sigmoid = (phi + 1) / 2;
d_phi_x = 2 .* sigmoid .* (1-sigmoid) * opts.sigscale;  % nbitsxN
d_L_x   = d_L_phi .* d_phi_x;

% 7. final
bot.dzdx = zeros(size(bot.x), 'single');
if onGPU, bot.dzdx = gpuArray(bot.dzdx); end
bot.dzdx(1, 1, :, :) = single(d_L_x);
end


function dHp = deriv_ent(p, minus1s)
% derivative of entropy
dHp = minus1s;
dHp(p>0) = dHp(p>0) - log2(p(p>0));
end
