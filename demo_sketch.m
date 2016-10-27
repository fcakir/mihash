function [resfn, dp] = demo_sketch(ftype, dataset, nbits, varargin)
addpath('sketch');

ip = inputParser;
ip.addParamValue('sketchSize', 200, @isscalar);
ip.addParamValue('batchSize', 50, @isscalar);
%ip.addParamValue('onlyFinal', 0, @isscalar);
ip.addParameter('methodID', 'sketch');
ip.KeepUnmatched = true;
ip.parse(varargin{:});
opts = ip.Results;
opts.identifier = sprintf('Ske%dBat%d', opts.sketchSize, opts.batchSize);
assert(opts.batchSize>=nbits, 'Sketching needs batchSize>=nbits');

% generic fields
opts = get_opts(opts, ftype, dataset, nbits, varargin{:});

% run demo
[resfn, dp] = demo(opts, @train_sketch, @test_sketch);
diary('off');
end
