function [resfn, dp] = demo_sketch(ftype, dataset, nbits, varargin)
% Implementation of the SketchHash method as described in: 
%
% C. Leng, J. Wu, J. Cheng, X. Bai and H. Lu
% "Online Sketching Hashing"
% Computer Vision and Pattern Recognition (CVPR) 2015
%
% INPUTS
% 	sketchSize - (int) size of the sketch matrix.
% 	 batchSize - (int) The batch size, i.e. size of the data chunk
% OUTPUTS
% 	resfn 	- (string) Path to the results file. see demo.m .
% 	dp 	- (string) Path to the diary which contains the command window text

addpath('SketchHash');

ip = inputParser;
ip.addParamValue('sketchSize', 200, @isscalar);
ip.addParamValue('batchSize', 50, @isscalar);
ip.KeepUnmatched = true;
ip.parse(varargin{:});
opts = ip.Results;
opts.identifier = sprintf('Ske%dBat%d', opts.sketchSize, opts.batchSize);
opts.methodID   = 'sketch';
assert(opts.batchSize>=nbits, 'Sketching needs batchSize>=nbits');

% generic fields
opts = get_opts(opts, ftype, dataset, nbits, varargin{:});

% run demo
[resfn, dp] = demo(opts, @train_sketchhash);
diary('off');
end
