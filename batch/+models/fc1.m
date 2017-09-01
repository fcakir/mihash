function [net, imageSize, normalize] = fc1(opts)

imageSize = 0;
normalize = [];

% only train FC layers on top of fc7 features
if opts.normalize
    lr = [1 0.1] ;
else
    lr = [1 1];
end
net.layers = {} ;

% FC1 (logits for each bit)
net.layers{end+1} = struct('type', 'conv', ...
    'name', 'fc1', ...
    'weights', {init_weights(1,4096,opts.nbits)}, ...
    'learningRate', lr, ...
    'stride', 1, ...
    'pad', 0) ;

% histogram-based loss layer
net.layers{end+1} = struct('type', 'custom', ...
    'name', 'loss', ...
    'opts', opts, ...
    'forward', str2func([opts.obj '_forward']), ...
    'backward', str2func([opts.obj, '_backward']));
net.layers{end}.precious = false;
net.layers{end}.weights = {};

% Meta parameters
net.meta.inputSize = [1 1 4096] ;

% Fill in default values
net = vl_simplenn_tidy(net) ;
end
