function resfn = demo_okh(ftype, dataset, nbits, varargin)

% OKH-specific fields
ip = inputParser;
ip.addParamValue('c', 0.1, @isscalar);
ip.addParamValue('alpha', 0.2, @isscalar);
for i = 1:2:length(varargin)-1
    % only parse defined fields, ignore others
    try
        ip.parse(varargin{i}, varargin{i+1});
    end
end
opts = ip.Results;
opts.identifier = sprintf('C%gA%g', opts.c, opts.alpha);
opts.localid = 'okh';

% generic fields
opts = get_opts(opts, ftype, dataset, nbits, varargin{:});  % set parameters

% run demo
resfn = demo(opts, @train_okh, @test_okh);
diary('off');
end
