classdef OSH
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
% additioanl information.
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
% Training routine for OSH method, see demo_osh.m .
%
% INPUTS
% 	Xtrain - (float) n x d matrix where n is number of points 
%       	         and d is the dimensionality 
%
% 	Ytrain - (int)   n x l matrix containing labels, for unsupervised datasets
% 			 might be empty, e.g., LabelMe.
%     thr_dist - (int)   For unlabelled datasets, corresponds to the distance 
%		         value to be used in determining whether two data instance
% 		         are neighbors. If their distance is smaller, then they are
% 		         considered neighbors.
%	       	         Given the standard setup, this threshold value
%		         is hard-wired to be compute from the 5th percentile 
% 		         distance value obtain through 2,000 training instance.
% 			 see load_gist.m . 
% 	prefix - (string) Prefix of the "checkpoint" files.
%   test_iters - (int)   A vector specifiying the checkpoints, see train.m .
%   trialNo    - (int)   Trial ID
%	opts   - (struct)Parameter structure.
%
% OUTPUTS
%  train_time  - (float) elapsed time in learning the hash mapping
%  update_time - (float) elapsed time in updating the hash table
%  res_time    - (float) elapsed time in maintaing the reservoir set
%  ht_updates  - (int)   total number of hash table updates performed
%  bit_computed_all - (int) total number of bit recomputations, see update_hash_table.m
% 
% NOTES
% 	W is d x b where d is the dimensionality 
%            and b is the bit length / # hash functions

properties
    ECOCs
    multi_labeled
    i_ecoc
    M_ecoc
    seenLabels
end

function W = init(obj, X, opts, bigM)
    % randomly generate candidate codewords, store in ECOCs
    if nargin < 4, bigM = 10000; end

    % NOTE ECOCs now is a BINARY (0/1) MATRIX!
    ECOCs = logical(zeros(bigM, opts.nbits));
    for t = 1:opts.nbits
        r = ones(bigM, 1);
        while (sum(r)==bigM || sum(r)==0)
            r = randi([0,1], bigM, 1);
        end
        ECOCs(:, t) = logical(r);
    end
    clear r

    d = size(X, 2);
    % LSH init
    W = randn(d, opts.nbits);
    W = W ./ repmat(diag(sqrt(W'*W))',d,1);

    multi_labeled = (size(Ytrain, 2) > 1);
    if multi_labeled, logInfo('Handling multi-labeled dataset'); end
    i_ecoc     = 1;  
    M_ecoc     = [];  
    seenLabels = [];
end

function W = train1batch(obj, W, X, Y, I, t, opts)
    % new training point
    ind = I(t);
    spoint = X(ind, :);
    slabel = Y(ind, :);
    
    % ---- Assign ECOC, etc ----
    if (~obj.multi_labeled && mod(slabel, 10) == 0) || ...
            (obj.multi_labeled && sum(slabel) > 0)
        % labeled (single- or multi-label): assign target code(s)
        isLabeled = true;
        num_labeled = num_labeled + 1;
        % TODO use a single struct for all ECOC stuff
        [target_codes, seenLabels, M_ecoc, i_ecoc] = find_target_codes(...
            slabel, seenLabels, M_ecoc, i_ecoc, ECOCs);
    else
        % unlabeled
        isLabeled = false;
        slabel = zeros(size(slabel));  % mark as unlabeled for subsequent functions
        num_unlabeled = num_unlabeled + 1;
    end

    % ---- hash function update ----
    % SGD. update W wrt. loss term(s)
    if isLabeled
        for c = 1:size(target_codes, 1)
            code = target_codes(c, :);
            W = sgd_update(W, spoint, code, opts.stepsize, opts.SGDBoost);
        end
    end
end


% -----------------------------------------------------------
% TODO prob make them member functions too
% SGD mini-batch update
function W = sgd_update(W, points, codes, stepsizes, SGDBoost)
    % input:
    %   W         - D*nbits matrix, each col is a hyperplane
    %   points    - n*D matrix, each row is a point
    %   codes     - n*nbits matrix, each row the corresp. target binary code
    %   stepsizes - SGD step sizes (1 per point) for current batch
    % output:
    %   updated W
    if SGDBoost == 0
        % no online boosting, hinge loss
        for i = 1:size(points, 1)
            xi = points(i, :);
            ci = codes(i, :);
            id = (xi * W .* ci < 1);  % logical indexing > find()
            n  = sum(id);
            if n > 0
                W(:,id) = W(:,id) + stepsizes(i)*(repmat(xi',[1 n])*diag(ci(id)));%*diag(bal(id));
            end
        end
    else
        % online boosting + exp loss
        for i = 1:size(points, 1)
            xi = points(i, :);
            ci = codes(i, :);
            st = stepsizes(i);
            for j = 1:size(W, 2)
                if j ~= 1
                    c1 = exp(-(ci(1:j-1)*(W(:,1:j-1)'*xi')));
                else
                    c1 = 1;
                end
                W(:,j) = W(:,j) - st * c1 * exp(-ci(j)*W(:,j)'*xi')*-ci(j)*xi';
            end
        end
    end
end

% -----------------------------------------------------------
% TODO prob make them member functions too
% find target codes for a new labeled example
function [target_codes, seenLabels, M_ecoc, i_ecoc] = find_target_codes(...
    slabel, seenLabels, M_ecoc, i_ecoc, ECOCs)
    assert(sum(slabel) ~= 0, ...
        'Error: finding target codes for unlabeled example');

    if numel(slabel) == 1
        % single-label dataset
        [ismem, ind] = ismember(slabel, seenLabels);
        if ismem == 0
            seenLabels = [seenLabels; slabel];
            % NOTE ECOCs now is a BINARY (0/1) MATRIX!
            M_ecoc = [M_ecoc; 2*ECOCs(i_ecoc,:)-1];
            ind    = i_ecoc;
            i_ecoc = i_ecoc + 1;
        end
    else
        % multi-label dataset
        if isempty(seenLabels)
            assert(isempty(M_ecoc));
            seenLabels = zeros(size(slabel));
            M_ecoc = zeros(numel(slabel), size(ECOCs, 2));
        end
        % find incoming labels that are unseen
        unseen = find((slabel==1) & (seenLabels==0));
        if ~isempty(unseen)
            for j = unseen
                % NOTE ECOCs now is a BINARY (0/1) MATRIX!
                M_ecoc(j, :) = 2*ECOCs(i_ecoc, :)-1;
                i_ecoc = i_ecoc + 1;
            end
            seenLabels(unseen) = 1;
        end
        ind = find(slabel==1);
    end

    % find/assign target codes
    target_codes = M_ecoc(ind, :);
end
