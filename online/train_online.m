function train_online(methodObj, run_trial, opts)
% Copyright (c) 2017, Fatih Cakir, Kun He, Saral Adel Bargal, Stan Sclaroff 
% All rights reserved.
% 
% If used for please cite the below paper:
%
% "MIHash: Online Hashing with Mutual Information", 
% Fatih Cakir*, Kun He*, Sarah Adel Bargal, Stan Sclaroff
% (* equal contribution)
% International Conference on Computer Vision (ICCV) 2017
% 
% Usage of code from authors not listed above might be subject
% to different licensing. Please check with the corresponding authors for
% information.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
% 
% 1. Redistributions of source code must retain the above copyright notice, this
%    list of conditions and the following disclaimer.
% 2. Redistributions in binary form must reproduce the above copyright notice,
%    this list of conditions and the following disclaimer in the documentation
%    and/or other materials provided with the distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
% ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
% ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
% ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
% 
% The views and conclusions contained in the software and documentation are those
% of the authors and should not be interpreted as representing official policies,
% either expressed or implied, of the FreeBSD Project.
%
%------------------------------------------------------------------------------
% This is the routine which calls different training subroutines based on the 
% hashing method. Separate trials are executed here and rudimentary statistics 
% are computed and displayed. 
%
% INPUTS
%    trainFunc - (func handle) Function handle determining which training routine
% 			       to call
%    run_trial - (vector)      Boolean vector specifying which trials to run.
% 			       if opts.override=0, previously ran trials are skipped.
%	opts   - (struct)      Parameter structure.
% OUTPUTS
% 	none

global Dataset

info = struct(...
    'train_time', [], ...      % time to learn the hash mapping
    'update_time', [], ...     % time to update the hash table
    'reservoir_time', [], ...  % time to update/maintain the reservoir
    'ht_updates', [], ...      % number of hash table updates performed
    'bit_recomp', []           % number of bit recomputations
    );
for n = fieldnames(info)
    info.(n{1}) = zeros(1, opts.ntrials);
end


% NOTE: if you have the Parallel Computing Toolbox, you can use parfor 
%       to run the trials in parallel
for t = 1:opts.ntrials
    if ~run_trial(t)
        logInfo('Trial %02d not required, skipped', t);
        continue;
    end
    logInfo('%s: random trial %d', opts.identifier, t);
    rng(opts.randseed+t, 'twister'); % fix randseed for reproducible results
    
    % randomly set test checkpoints
    test_iters      = zeros(1, opts.ntests);
    test_iters(1)   = 1;
    test_iters(end) = num_iters;
    interval = round(num_iters/(opts.ntests-1));
    for i = 1:opts.ntests-2
        iter = interval*i + randi([1 round(interval/3)]) - round(interval/6);
        test_iters(i+1) = iter;
    end
    prefix = sprintf('trial%d', t);
    
    % train hash functions
    % TODO train_one_method
    info = train_one_method(methodObj, Dataset, prefix, test_iters, opts);
end

% TODO use info struct
logInfo(' Training time (total): %.2f +/- %.2f', mean(train_time), std(train_time));
logInfo('HT update time (total): %.2f +/- %.2f', mean(update_time), std(update_time));
logInfo('Reservoir time (total): %.2f +/- %.2f', mean(resservoir_time), std(resservoir_time));
logInfo('');
logInfo('Hash Table Updates (per): %.4g +/- %.4g', mean(ht_updates), std(ht_updates));
logInfo('Bit Recomputations (per): %.4g +/- %.4g', mean(bit_recomp), std(bit_recomp));
end


function train_one_method()
end
