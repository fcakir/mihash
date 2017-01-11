function [resfn, dp] = demo_mutualinfo(ftype, dataset, nbits, varargin)
% PARAMS
%  ftype (string) from {'gist', 'cnn'}
%  dataset (string) from {'cifar', 'sun','nus'}
%  nbits (integer) is length of binary code
%  varargin: see get_opts.m for details

% get OSH-specific fields first
ip = inputParser;
ip.addParamValue('no_bins', 16, @isscalar);
ip.addParamValue('stepsize', 1, @isscalar);
ip.addParamValue('sigmf_p', [1 0], @isnumeric);
ip.addParameter('methodID', 'mutual_info');
ip.KeepUnmatched = true;
ip.parse(varargin{:});
opts = ip.Results;
opts.identifier = sprintf('NoBins%dS%g', opts.no_bins, opts.stepsize);
opts.batchSize  = 1;  % hard-coded

% get generic fields
opts = get_opts(opts, ftype, dataset, nbits, varargin{:});  % set parameters

% run demo
[resfn, dp] = demo(opts, @train_mutualinfo, @test_osh);
diary('off');
end
