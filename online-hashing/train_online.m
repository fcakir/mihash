function info = train_online(methodObj, Dataset, trial, res_file, test_iters, opts)
% Copyright (c) 2017, Fatih Cakir, Kun He, Saral Adel Bargal, Stan Sclaroff 
% All rights reserved.
% 
% If used for academic purposes please cite the below paper:
%
% "MIHash: Online Hashing with Mutual Information", 
% Fatih Cakir*, Kun He*, Sarah Adel Bargal, Stan Sclaroff
% (* equal contribution)
% International Conference on Computer Vision (ICCV) 2017
% 
% Usage of code from authors not listed above might be subject
% to different licensing. Please check with the corresponding authors for
% additional information.
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
% Training routine for online hashing
%
% INPUTS
%    methodObj - (object)
%      Dataset - (struct) 
% 	 trial - (int) trial number
%   test_iters - (int) A vector specifiying the checkpoints, see train.m .
%         opts - (struct) Parameter structure.
%
% OUTPUTS
%  train_time  - (float) elapsed time in learning the hash mapping
%  update_time - (float) elapsed time in updating the hash table
%  reserv_time - (float) elapsed time in maintaing the reservoir set
%  ht_updates  - (int)   total number of hash table updates performed
%  bits_computed - (int) total number of bit recomputations, see update_hash_table.m
% 

%%%%%%%%%%%%%%%%%%%%%%% INIT %%%%%%%%%%%%%%%%%%%%%%%
Xtrain = Dataset.Xtrain;
Ytrain = Dataset.Ytrain;

% set up reservoir
reservoir = [];
if opts.reservoirSize > 0
    reservoir.size = 0;
    reservoir.PQ   = [];  % priority queue
    reservoir.H    = [];  % hash table for the reservoir
    reservoir.X    = zeros(0, size(Xtrain, 2));
    reservoir.Y    = zeros(0, size(Ytrain, 2));
end

% hash mapping & hash table
[W, reservoir, methodObj] = methodObj.init(reservoir, Xtrain, Ytrain, opts);
H = methodObj.encode(W, Xtrain, false);
Wsnapshot = W;  % snapshot hash mapping

% order training examples
ntrain = size(Xtrain, 1);
assert(opts.numTrain<=ntrain, sprintf('opts.numTrain > %d!', ntrain));
trainInd = [];
for e = 1:opts.epoch
    trainInd = [trainInd, randperm(ntrain, opts.numTrain)];
end
numTrainTotal = numel(trainInd);

info = [];
info.H            = [];
info.W            = [];
info.Wsnapshot    = [];
info.update_iters = [];
info.bit_recomp   = 0;
info.time_train   = 0;
info.time_update  = 0;
info.time_reserv  = 0;

iterdir = fullfile(opts.dirs.exp, sprintf('trial%d_iter', trial));
if ~exist(iterdir), mkdir(iterdir); end
%%%%%%%%%%%%%%%%%%%%%%% INIT %%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%% STREAMING BEGINS! %%%%%%%%%%%%%%%%%%%%%%%
opts.num_iters = ceil(numTrainTotal / opts.batchSize);
logInfo('%s: %d train_iters', opts.identifier, opts.num_iters);

update_table = false;
for iter = 1:opts.num_iters
    t_ = tic;
    [W, batch] = methodObj.train1batch(W, reservoir, Xtrain, Ytrain, ...
        trainInd, iter, opts);
    info.time_train = info.time_train + toc(t_);

    % ---- reservoir update & compute new reservoir hash table ----
    t_ = tic;
    Hres_new = [];
    if opts.reservoirSize > 0
        % update reservoir
        [reservoir, update_ind] = update_reservoir(reservoir, ...
            Xtrain(batch, :), Ytrain(batch, :), ...
            opts.reservoirSize, opts.unsupervised);
        % update reservoir hash table (with snapshot)
        Hres = methodObj.encode(Wsnapshot, reservoir.X, true);
        reservoir.H(update_ind, :) = Hres(update_ind, :);
    end

    % ---- determine whether to update or not ----
    if ~mod(iter*opts.batchSize, opts.updateInterval)
        % new reservoir hash table (with new W)
        Hres_new = methodObj.encode(W, reservoir.X, true);
        update_table = trigger_update(iter, Wsnapshot, W, reservoir, ...
            Hres_new, opts);
    end
    info.time_reserv = info.time_reserv + toc(t_);

    % ---- hash table update, etc ----
    if update_table
        Wsnapshot = W;  % update snapshot
        info.update_iters = [info.update_iters, iter];
        if opts.reservoirSize > 0
            reservoir.H = Hres_new;
        end

        % recompute hash table
        % NOTE: We only record time for matrix multiplication here as the data 
        %       (Xtrain) is completely loaded in memory. This is not necessarily 
        %       the case with large databases, where disk I/O would probably be 
        %       involved in the hash table recomputation.
        t_ = tic;
        H  = methodObj.encode(Wsnapshot, Xtrain, false);
        info.bit_recomp  = info.bit_recomp + prod(size(H));
        info.time_update = info.time_update + toc(t_);
        update_table = false;
    end

    % ---- CHECKPOINT: save intermediate model ----
    if ismember(iter, test_iters)
        info.params = methodObj.get_params();
        info.Wsnapshot = Wsnapshot;
        info.W = W;
        info.H = H;
        F = sprintf('%s/%d.mat', iterdir, iter);
        save(F, '-struct', 'info');

        logInfo('');
        logInfo('[T%d] CHECKPOINT @ iter %d/%d (batchSize %d)', trial, iter, ...
            opts.num_iters, opts.batchSize);
        logInfo('[%s] %s', opts.methodID, opts.identifier);
        logInfo('W %.2fs, HT %.2fs (%d updates), Res %.2fs. #BR: %.3g', ...
            info.time_train, info.time_update, numel(info.update_iters), ...
            info.time_reserv, info.bit_recomp);
        logInfo('');
    end
end
%%%%%%%%%%%%%%%%%%%%%%% STREAMING ENDED! %%%%%%%%%%%%%%%%%%%%%%%

% finalize info
info.ht_updates = numel(info.update_iters);
info.test_iters = test_iters;
info.params = methodObj.get_params();
logInfo('HT updates: %d, bits computed: %d', info.ht_updates, info.bit_recomp);

% save
save(res_file, '-struct', 'info');
logInfo('[Trial %d] Saved: %s\n', trial, res_file);

% rmfields
info = rmfield(info, 'params');
info = rmfield(info, 'Wsnapshot');
info = rmfield(info, 'W');
info = rmfield(info, 'H');
end
