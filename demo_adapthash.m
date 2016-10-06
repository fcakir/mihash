function resfn = demo_adapthash(ftype, dataset, nbits, varargin)

% AdaptHash-specific fields
ip = inputParser;
ip.addRequired('localid', @isstr);
ip.addParamValue('alpha', 0.9, @isscalar);
ip.addParamValue('beta', 1e-2, @isscalar);
ip.addParamValue('stepsize', 1, @isscalar);
ip.parse('adaptkeyboard
for i = 1:2:length(varargin)-1
    % only parse defined fields, ignore others
    try
        ip.parse(varargin{i}, varargin{i+1})
    end
end
%try, ip.parse(varargin{:}); end
opts = ip.Results
opts.identifier = sprintf('A%gB%gS%g', opts.alpha, opts.beta, opts.stepsize);

% generic fields
opts = get_opts(opts, ftype, dataset, nbits, varargin{:});  % set parameters

% run demo
resfn = demo(opts, @train_adapthash, @test_adapthash);
diary('off');
end
