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
    instFeatAvePre
    instFeatSkc
    instCntSeen
end

methods
    function [W, R, obj] = init(obj, R, X, Y, opts)
        d = size(X, 2);  % feature dim
        assert(opts.sketchSize<=d, sprintf('Need sketchSize<=d(%d)', d));

        if 0
            % original init for SketchHash, performed worse
            W = rand(d, opts.nbits) - 0.5;
        else
            % LSH init
            W = randn(d, opts.nbits);
            W = W ./ repmat(diag(sqrt(W'*W))',d,1);
        end

        obj.instFeatAvePre = zeros(1, d);  % mean vector
        obj.instFeatSkc    = [];           % sketch matrix
        obj.instCntSeen    = 0;
        logInfo('%d batches of size %d, sketchSize %d', ...
            ceil(opts.numTrain/batchsize), opts.batchSize, opts.sketchSize);
    end


    function [W, ind] = train1batch(obj, W, R, X, Y, I, t, opts)

        %%%%%%%%%% LOAD BATCH DATA - BELOW %%%%%%%%%%
        ind = (t-1)*opts.batchSize + (1:opts.batchSize);
        ind = I(ind);
        instFeatInBatch = X(ind, :);

        instCntInBatch = size(instFeatInBatch, 1);
        %%%%%%%%%% LOAD BATCH DATA - ABOVE %%%%%%%%%%


        %%%%%%%%%% UPDATE HASHING FUNCTION - BELOW %%%%%%%%%%
        % calculate current mean feature vector
        instFeatAveCur = mean(instFeatInBatch, 1);

        % sketech current training batch
        if t == 1
            instFeatToSkc = bsxfun(@minus, instFeatInBatch, instFeatAveCur);
        else
            instFeatCmps = sqrt(obj.instCntSeen * instCntInBatch / ...
                (obj.instCntSeen + instCntInBatch)) * (instFeatAveCur - obj.instFeatAvePre);
            instFeatToSkc = [bsxfun(@minus, instFeatInBatch, instFeatAveCur); ...
                instFeatCmps];
        end
        obj.instFeatSkc = obj.MatrixSketch_Incr(obj.instFeatSkc, instFeatToSkc, ...
            opts.sketchSize);

        % update mean feature vector and instance counter
        obj.instFeatAvePre = (obj.instFeatAvePre * obj.instCntSeen + ...
            instFeatAveCur * instCntInBatch) / (obj.instCntSeen + instCntInBatch);
        obj.instCntSeen = obj.instCntSeen + instCntInBatch;

        % compute QR decomposition of the sketched matrix
        [q, r] = qr(obj.instFeatSkc', 0);
        [u, ~, ~] = svd(r, 'econ');
        v = q * u;

        % obtain the original projection matrix
        hashProjMatOrg = v(:, 1 : bits);

        % use random rotation
        R = orth(randn(bits));

        % update hashing function
        W = hashProjMatOrg * R;
        %%%%%%%%%% UPDATE HASHING FUNCTION - ABOVE %%%%%%%%%%
    end


    function B = MatrixSketch_Incr(obj, B, A, l)
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

    
    function H = encode(obj, W, X, isTest)
        X = bsxfun(@minus, X, obj.instFeatAvePre);
        H = (X * W) > 0;
    end

    function P = get_params(obj)
        P = [];
        P.instFeatAvePre = obj.instFeatAvePre;
    end

end % methods

end % classdef
