function [resfn, dp] = demo_okh(ftype, dataset, nbits, varargin)
% Implementation of OKH as described in: 
%
% L. K. Huang, Q. Y. Yang and W. S. Zheng
% "Online Hashing"
% International Joint Conference on Artificial Intelligence (IJCAI) 2013
%
% INPUTS
%	c 	 - (float) Parameter C as in Alg. 1 of OKH. 
% 	alpha	 - (float) \alpha as in Eq. 3 of OKH
% OUTPUTS
% 	resfn 	- (string) Path to the results file. see demo.m .
% 	dp 	- (string) Path to the diary which contains the command window text

% OKH-specific fields
ip = inputParser;
ip.addParamValue('c', 0.1, @isscalar);
ip.addParamValue('alpha', 0.2, @isscalar);
ip.KeepUnmatched = true;
ip.parse(varargin{:});
opts = ip.Results;
opts.identifier = sprintf('C%gA%g', opts.c, opts.alpha);
opts.methodID   = 'okh';
opts.batchSize  = 2;  % hard-coded; pair supervision

% generic fields
opts = get_opts(opts, ftype, dataset, nbits, varargin{:});

% run demo
[resfn, dp] = demo(opts, @train_okh);
diary('off');
end
