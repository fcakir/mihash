classdef SketchHash
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

properties
    kInstFeatDimCnt
    numUseToTrain
    batchsize
    batchCnt
    instCntSeen
    instFeatAvePre
    instFeatSkc
end

methods
    function W = init(obj, X, opts)
        kInstFeatDimCnt = size(X, 2);  % feature dim
        bits = opts.nbits;
        assert(opts.sketchSize <= kInstFeatDimCnt, ...
            sprintf('Sketching needs sketchSize<=d(%d)', kInstFeatDimCnt));

        % initialize hash functions & table
        if 0
            % original init for SketchHash, which performed worse
            W = rand(kInstFeatDimCnt, bits) - 0.5;
        else
            % LSH init
            d = kInstFeatDimCnt;
            W = randn(d, bits);
            W = W ./ repmat(diag(sqrt(W'*W))',d,1);
        end
        % prepare to run online sketching hashing
        if opts.noTrainingPoints > 0
            numUseToTrain = opts.noTrainingPoints;
        else
            numUseToTrain = size(X, 1);
        end
        batchsize      = opts.batchSize;
        batchCnt       = ceil(numUseToTrain/batchsize);
        instCntSeen    = 0;
        instFeatAvePre = zeros(1, kInstFeatDimCnt);  % mean vector
        instFeatSkc    = [];
        logInfo('%d batches of size %d, sketchSize=%d', ...
            batchCnt, batchsize, opts.sketchSize);
    end


    function [W, ind] = train1batch(obj, W, X, Y, I, t, opts)
        % TODO use I
        batchInd = t;

        %%%%%%%%%% LOAD BATCH DATA - BELOW %%%%%%%%%%
        ind = (batchInd-1)*batchsize + 1 : min(batchInd*batchsize, numUseToTrain);
        ind = I(ind);
        instFeatInBatch = X(ind, :);

        instCntInBatch = size(instFeatInBatch, 1);
        %%%%%%%%%% LOAD BATCH DATA - ABOVE %%%%%%%%%%


        %%%%%%%%%% UPDATE HASHING FUNCTION - BELOW %%%%%%%%%%
        tic;

        % calculate current mean feature vector
        instFeatAveCur = mean(instFeatInBatch, 1);

        % sketech current training batch
        if batchInd == 1
            instFeatToSkc = bsxfun(@minus, instFeatInBatch, instFeatAveCur);
        else
            instFeatCmps = sqrt(instCntSeen * instCntInBatch / ...
                (instCntSeen + instCntInBatch)) * (instFeatAveCur - instFeatAvePre);
            instFeatToSkc = [bsxfun(@minus, instFeatInBatch, instFeatAveCur); ...
                instFeatCmps];
        end
        instFeatSkc = MatrixSketch_Incr(instFeatSkc, instFeatToSkc, opts.sketchSize);

        % update mean feature vector and instance counter
        instFeatAvePre = (instFeatAvePre * instCntSeen + ...
            instFeatAveCur * instCntInBatch) / (instCntSeen + instCntInBatch);
        instCntSeen = instCntSeen + instCntInBatch;

        % compute QR decomposition of the sketched matrix
        [q, r] = qr(instFeatSkc', 0);
        [u, ~, ~] = svd(r, 'econ');
        v = q * u;

        % obtain the original projection matrix
        hashProjMatOrg = v(:, 1 : bits);

        % use random rotation
        R = orth(randn(bits));

        % update hashing function
        W = hashProjMatOrg * R;

        train_time = train_time + toc;
        %%%%%%%%%% UPDATE HASHING FUNCTION - ABOVE %%%%%%%%%%
    end


    function B = MatrixSketch_Incr(B, A, l)
        if mod(l,2) ~= 0
            error('l should be an even number...')
        end
        ind = l/2;
        [n, ~] = size(A); % n: number of samples; m: dimension
        numNonzeroRows = numel(sum(B .^ 2, 2) > 0); % number of non-zero rows
        for i = 1 : n
            if numNonzeroRows < l
                numNonzeroRows = numNonzeroRows + 1; %disp(numNonzeroRows);
                B(numNonzeroRows,:) = A(i,:);
            else
                [q, r] = qr(B', 0);
                [u, sigma, ~] = svd(r, 'econ');
                v = q * u;
                sigmaSquare = sigma .^ 2;
                sigmaSquareDiag = diag(sigmaSquare);
                theta = sigmaSquareDiag(ind + 1);
                sigmaHat = sqrt(max((sigmaSquare - eye(l) * theta),0));
                B = sigmaHat * v';

                numNonzeroRows = ind;

                numNonzeroRows = numNonzeroRows + 1; 
                B(numNonzeroRows,:) = A(i,:);
            end
        end
    end

end % methods

end % classdef
