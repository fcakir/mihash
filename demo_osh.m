function resfn = demo_osh(ftype, dataset, nbits, varargin)
% PARAMS
%  ftype (string) from {'gist', 'cnn'}
%  dataset (string) from {'cifar', 'sun','nus'}
%  nbits (integer) is length of binary code
%  varargin: see get_opts.m for details

% get OSH-specific fields first
ip = inputParser;
ip.addParamValue('stepsize', 0.1, @isscalar);
ip.addParamValue('SGDBoost', 1, @isscalar);
ip.addParameter('methodID', '');
ip.KeepUnmatched = true;
ip.parse(varargin{:});
opts = ip.Results;
opts.identifier = sprintf('B%dS%g', opts.SGDBoost, opts.stepsize);
opts.batchSize  = 1;  % hard-coded

% get generic fields
opts = get_opts(opts, ftype, dataset, nbits, varargin{:});  % set parameters

% run demo
resfn = demo(opts, @train_osh, @test_osh);
diary('off');
end
