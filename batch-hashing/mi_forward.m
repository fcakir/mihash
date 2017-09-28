function [top] = mi_forward(layer, bot, top)
% vectorized implementation of MIHash (minibatch)
% forward pass

Y = squeeze(layer.class); % Nx1
X = squeeze(bot.x);  % 1x1xBxN -> BxN, raw scores (logits) for each bit
[nbits, N] = size(X);
assert(size(Y, 1) == N);

opts = layer.opts;
onGPU = numel(opts.gpus) > 0;

% get NxN affinity matrix
Aff = affinity(X', X', Y, Y, opts);
Xp  = logical(Aff - diag(diag(Aff)));
Xn  = ~Aff;
if onGPU
    Xp = gpuArray(Xp);
    Xn = gpuArray(Xn);
end

% histogram params
no_bins = opts.nbins;
histC = linspace(0, nbits, no_bins+1);
histD = nbits / no_bins;

% compute distances from hash codes
phi = 2*sigmoid(X, opts.sigscale) - 1;  % RELAXED hash codes to interval [-1, 1]
hdist = (nbits - phi' * phi)/2;   % NxN pairwise dist matrix

% estimate discrete distributions
prCp = sum(Xp, 2) ./ (N-1);
prCn = 1 - prCp;
pDCp = zeros(N, no_bins+1);
pDCn = zeros(N, no_bins+1);
if onGPU
    pDCp = gpuArray(pDCp);
    pDCn = gpuArray(pDCn);
end

% new version, better when L<N
for l = 1:no_bins+1
    pulse = triPulse(hdist, histC(l), histD);  % NxN
    pDCp(:, l) = sum(pulse .* Xp, 2);
    pDCn(:, l) = sum(pulse .* Xn, 2);
end

% pD
pD = (pDCp + pDCn) ./ (N-1);
% normalize
sum_pDCp = sum(pDCp, 2);  nz_p = sum_pDCp > 0;
sum_pDCn = sum(pDCn, 2);  nz_n = sum_pDCn > 0;
pDCp(nz_p, :) = bsxfun(@rdivide, pDCp(nz_p, :), sum_pDCp(nz_p));
pDCn(nz_n, :) = bsxfun(@rdivide, pDCn(nz_n, :), sum_pDCn(nz_n));

% compute entropies
y0 = zeros(size(pD));
if onGPU, y0 = gpuArray(y0); end
ent_D   = ent(pD, y0);  % H(D)
ent_D_C = prCp .* ent(pDCp, y0) + prCn .* ent(pDCn, y0);  % H(D|C)

% loss
top.x = sum(single(ent_D - ent_D_C));  % display MI, but minimize -MI
top.aux = [];
top.aux.phi   = phi;
top.aux.hdist = hdist;
top.aux.histC = histC;
top.aux.histD = histD;
top.aux.Xp    = Xp;
top.aux.Xn    = Xn;
top.aux.prCp  = prCp;
top.aux.prCn  = prCn;
top.aux.pDCp  = pDCp;
top.aux.pDCn  = pDCn;
top.aux.pD    = pD;
end


function H = ent(p, y0)
logp = y0;
logp(p>0) = log2(p(p>0));
H = -sum(p .* logp, 2);
end
