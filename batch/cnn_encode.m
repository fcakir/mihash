function H = cnn_encode(net, batchFunc, imdb, ids, opts, noLossLayer)
if noLossLayer
    layerOffset = 0;
else
    layerOffset = 1;
end
fprintf('Testing network... '); tic;
batch_size = opts.batchSize;
onGPU = ~isempty(opts.gpus);

H = zeros(opts.nbits, length(ids), 'single');
for t = 1:batch_size:length(ids)
    ed = min(t+batch_size-1, length(ids));
    [data, labels] = batchFunc(imdb, ids(t:ed));
    net.layers{end}.class = labels;
    if onGPU
        data = gpuArray(data); 
        res = vl_simplenn(net, data, [], [], 'mode', 'test');
        rex = squeeze(gather(res(end-layerOffset).x));
    else
        res = vl_simplenn(net, data, [], [], 'mode', 'test');
        rex = squeeze(res(end-layerOffset).x);
    end
    H(:, t:ed) = single(rex > 0);
end
toc;
end
