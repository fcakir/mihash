function [V, D] = CalcEigsDscd(X,n)
    [V, D]= eig(X);
    %[V, D]= eigs(X,n);
    D = diag(D);
    [D,Dindice] = sort(D,'descend');
    D = diag(D);
    V = V(:,Dindice);
end