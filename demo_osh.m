function [resfn, dp] = demo_osh(ftype, dataset, nbits, varargin)
% PARAMS
%  ftype (string) from {'gist', 'cnn'}
%  dataset (string) from {'cifar', 'sun','nus'}
%  nbits (integer) is length of binary code
%  varargin: see get_opts.m for details

% get OSH-specific fields first
ip = inputParser;
ip.addParamValue('stepsize', 0.1, @isscalar);
ip.addParamValue('SGDBoost', 1, @isscalar);
ip.addParamValue('learn_ecoc', 0, @isscalar);
ip.addParamValue('cluster_size', 2000, @isscalar);
ip.addParameter('methodID', 'osh');
ip.KeepUnmatched = true;
ip.parse(varargin{:});
opts = ip.Results;
opts.identifier = sprintf('B%d_S%g_LECOC%g_ClustSize%g', opts.SGDBoost, opts.stepsize, ...
	opts.learn_ecoc, opts.cluster_size);
opts.batchSize  = 1;  % hard-coded

% get generic fields
opts = get_opts(opts, ftype, dataset, nbits, varargin{:});  % set parameters

% run demo
[resfn, dp] = demo(opts, @train_osh, @test_osh);
diary('off');
end
