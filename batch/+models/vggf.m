function [net, imageSize, normalize] = vgg_f(opts)

imageSize = 224;
normalize = false;
lr = [1 1];

% finetune VGG-16
net = load(fullfile(opts.localDir, 'models', 'imagenet-vgg-f.mat'));
net.layers(end) = [];
net.layers(end) = [];

if 0
    % insert dropout layers (removed in deploy model)
    assert(strcmp(net.layers{end}.name, 'relu7'));
    net.layers(end:end+1) = net.layers(end-1:end);
    net.layers{end-2} = struct('type', 'dropout', 'name', 'drop6', ...
        'rate', 0.5);
    net.layers{end+1} = struct('type', 'dropout', 'name', 'drop7', ...
        'rate', 0.5);
end

% FC layer
net.layers{end+1} = struct('type', 'conv', 'name', 'logits', ...
    'weights', {init_weights(1, 4096, opts.nbits)}, ...
    'learningRate', lr, ...
    'stride', 1, 'pad', 0);

% loss layer
net.layers{end+1} = struct('type', 'custom', 'name', 'loss', ...
    'opts', opts, 'weights', [], 'precious', false, ...
    'forward', str2func([opts.obj '_forward']), ...
    'backward', str2func([opts.obj, '_backward']));

cellfun(@(x) x.name, net.layers, 'uniform', 0)
net = vl_simplenn_tidy(net);
end