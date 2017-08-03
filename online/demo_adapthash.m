function [resfn, dp] = demo_adapthash(ftype, dataset, nbits, varargin)
% Implementation of AdaptHash as described in: 
%
% F. Cakir, S. Sclaroff
% "Adaptive Hashing for Fast Similarity Search"
% International Conference on Computer Vision (ICCV) 2015
%
% INPUTS
%	alpha 	 - (float) [0, 1] \alpha as in Alg. 1 of AdaptHash. 
% 	beta 	 - (float) \lambda as in Alg. 1
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
ip.addParamValue('alpha', 0.9, @isscalar);
ip.addParamValue('beta', 1e-2, @isscalar);
ip.addParamValue('stepsize', 1, @isscalar);
ip.KeepUnmatched = true;
ip.parse(varargin{:});
opts = ip.Results;
opts.methodID   = 'adapt';
opts.identifier = sprintf('A%gB%gS%g', opts.alpha, opts.beta, opts.stepsize);
opts.batchSize  = 2;  % hard-coded; pair supervision

% generic fields
opts = get_opts(opts, ftype, dataset, nbits, varargin{:});  % set parameters

% run demo
[resfn, dp] = demo(opts, @train_adapthash);
diary('off');
end
