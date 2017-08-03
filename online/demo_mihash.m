function [resfn, dp] = demo_mihash(ftype, dataset, nbits, varargin)
% Implementation of AdaptHash as described in: 
%
% Fatih Cakir*, Kun He*, Sarah Adel Bargal, Stan Sclaroff 
% "MIHash: Online Hashing with Mutual Information", (*equal contribution).
% International Conference on Computer Vision (ICCV) 2015
%
% INPUTS
%	no_bins  - (int in [1, nbits]) specifies the number of bins of the 
%	           histogram (K in Section 3.2)
% 	stepsize - (float) The learning rate.
% 	decay    - (float) Decay parameter for learning rate. 
% 	sigscale - (10) Sigmoid function to smooth the sgn of the hash function,
% 	           used as second argument to sigmf.m, see Section 3.2
%         initRS - (int) Initial reservoir size. Must be a positive value. 
%                  >=500 is recommended. 
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
ip.addParamValue('sigscale', 10, @isscalar);
ip.addParamValue('initRS', 500, @isscalar); % initial size of reservoir
ip.KeepUnmatched = true;
ip.parse(varargin{:});
opts = ip.Results;
opts.methodID   = 'mihash';
opts.identifier = sprintf('Bins%dSig%g_Step%gDecay%g_InitRS%g', opts.no_bins, ...
    opts.sigscale, opts.stepsize, opts.decay, opts.initRS);
opts.batchSize  = 1;  % hard-coded

% get generic fields
opts = get_opts(opts, ftype, dataset, nbits, varargin{:});  % set parameters

% run demo
[resfn, dp] = demo(opts, @train_mihash);
diary('off');
end
