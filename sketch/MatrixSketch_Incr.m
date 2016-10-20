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
            
%             [~, r] = qr(B);
%             [~, sigma, v] = svd(r, 'econ');
%             sigmaSquare = sigma .^ 2;
%             sigmaSquareDiag = diag(sigmaSquare);
%             theta = sigmaSquareDiag(ind + 1);
%             sigmaHat = sqrt(max((sigmaSquare - eye(l) * theta),0));
%             B = sigmaHat * v';
            
%             [~,sigma,V] = svd(B,'econ');
%             sigmaSquare = sigma.^2;
%             sigmaSquareDiag = diag(sigmaSquare);
%             theda = sigmaSquareDiag(ind + 1);
%             sigmaHat = sqrt(max((sigmaSquare-eye(l)*theda),0));
%             B = sigmaHat * V';
            
            numNonzeroRows = ind;
            
            numNonzeroRows = numNonzeroRows + 1; 
            B(numNonzeroRows,:) = A(i,:);
        end
    end
end
