function resfn = demo_adapthash(ftype, dataset, nbits, varargin)

% AdaptHash-specific fields
ip = inputParser;
ip.addParamValue('alpha', 0.9, @isscalar);
ip.addParamValue('beta', 1e-2, @isscalar);
ip.addParamValue('stepsize', 1, @isscalar);
for i = 1:2:length(varargin)-1
    % only parse defined fields, ignore others
    try
        ip.parse(varargin{i}, varargin{i+1});
    end
end
opts = ip.Results;
opts.identifier = sprintf('A%gB%gS%g', opts.alpha, opts.beta, opts.stepsize);
opts.localid = 'adapthash';

% generic fields
opts = get_opts(opts, ftype, dataset, nbits, varargin{:});  % set parameters

% run demo
resfn = demo(opts, @train_adapthash, @test_adapthash);
diary('off');
end
