function [resfn, dp] = demo_adapthash(ftype, dataset, nbits, varargin)

% AdaptHash-specific fields
ip = inputParser;
ip.addParamValue('alpha', 0.9, @isscalar);
ip.addParamValue('beta', 1e-2, @isscalar);
ip.addParamValue('stepsize', 1, @isscalar);
ip.addParameter('methodID', 'adapt');
ip.KeepUnmatched = true;
ip.parse(varargin{:});
opts = ip.Results;
opts.identifier = sprintf('A%gB%gS%g', opts.alpha, opts.beta, opts.stepsize);
opts.batchSize  = 2;  % hard-coded; pair supervision

% generic fields
opts = get_opts(opts, ftype, dataset, nbits, varargin{:});  % set parameters

% run demo
[resfn, dp] = demo(opts, @train_adapthash, @test_osh);
diary('off');
end
