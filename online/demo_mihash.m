function [resfn, dp] = demo_mihash(ftype, dataset, nbits, varargin)
% Implementation of AdaptHash as described in: 
%
% Fatih Cakir*, Kun He*, Sarah Adel Bargal, Stan Sclaroff 
% "MIHash: Online Hashing with Mutual Information", (*equal contribution).
% International Conference on Computer Vision (ICCV) 2015
%
% INPUTS
%	no_bins  - (int) [1, code length] specifies the number of bins of the histogram, 
% 									  K in Section 3.2
% 	stepsize - (float) The learning rate.
% 	decay    - (float) Decay parameter for learning rate. 
% 	sigmf_p  - ([1 0]) Sigmoid function to smooth the sgn of the hash function,
% 					   used as second argument to sigmf.m, see Section 3.2
%   init_r_size - (int) Initial reservoir size. Must be a positive value. > 500
% 						is recommended. 
% OUTPUTS
% 	resfn 	- (string) Path to the results file. see demo.m .
% 	dp 	- (string) Path to the diary which contains the command window text
%  ftype (string) from {'gist', 'cnn'}
%  dataset (string) from {'cifar', 'sun','nus'}
%  nbits (integer) is length of binary code
%  varargin: see get_opts.m for details

% get MIHash-specific fields first
ip = inputParser;
ip.addParamValue('no_bins', 16, @isscalar);
ip.addParamValue('stepsize', 1, @isscalar);
ip.addParamValue('decay', 0, @isscalar);
ip.addParamValue('sigmf_p', [1 0], @isnumeric);
ip.addParamValue('init_r_size', 500, @isscalar); % initial size of reservoir
ip.KeepUnmatched = true;
ip.parse(varargin{:});
opts = ip.Results;
opts.identifier = sprintf('NoBins%d_StepSize%g_Decay%g_InitRSize%g_SGMFP%g-%g', opts.no_bins, ...
        opts.stepsize, opts.decay, opts.init_r_size, opts.sigmf_p(1), opts.sigmf_p(2));
opts.methodID  = 'mihash';
opts.batchSize = 1;  % hard-coded

% get generic fields
opts = get_opts(opts, ftype, dataset, nbits, varargin{:});  % set parameters

% run demo
[resfn, dp] = demo(opts, @train_mihash);
diary('off');
end
