function [resfn, dp] = demo_osh(ftype, dataset, nbits, varargin)
% Implementation of the OSH method as described in: 
%
% F. Cakir, S. Sclaroff
% "Online Supervised Hashing"
% International Conference on Image Processing (ICIP) 2015
%
% F. Cakir, S. A. Bargal, S. Sclaroff
% "Online Supervised Hashing"
% Computer Vision and Image Understanding (CVIU) 2016
%
% INPUTS
% 	stepsize - (float) The learning rate.
% 	SGDBoost - (int)   Choices are {0, 1}.  SGDBoost=1 corresponds to do the 
% 			   online boosting formulation with exponential loss as 
% 			   described in the above papers. SGDBoost=0, corresponds
% 			   to a hinge loss formulation without the online boosting 
% 			   approach. SGDBoost=0 typically works better.
% OUTPUTS
% 	resfn 	- (string) Path to the results file. see demo.m .
% 	dp 	- (string) Path to the diary which contains the command window text

ip = inputParser;
ip.addParamValue('stepsize', 0.1, @isscalar);
ip.addParamValue('SGDBoost', 1, @isscalar);
ip.KeepUnmatched = true;
ip.parse(varargin{:});
opts = ip.Results;
opts.methodID   = 'osh';  % hard-coded
opts.identifier = sprintf('B%d_S%g', opts.SGDBoost, opts.stepsize);
opts.batchSize  = 1;      % hard-coded

% get generic fields
opts = get_opts(opts, ftype, dataset, nbits, varargin{:});  % set parameters

% run demo
[resfn, dp] = demo(opts, @train_osh);
diary('off');
end
