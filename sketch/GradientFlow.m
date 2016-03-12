function Q = GradientFlow(Lambda)
% Input: matrix Lambda. Output: orthogonal matrix Q
    global Xf;
    n = size(Lambda,1);
    a1 = trace(Lambda)/n;
    a = a1*ones(1,n);
    %random orthogonal matrix U
    R = randn(n,n);
    [U, ~, ~] = svd(R);
    X0 = U*Lambda*U';
    X0 = reshape(X0,length(X0(:)),1);
    options = odeset('Refine',1,'OutputFcn',@(t,X,flag)check(t,X,flag,a));
    ode113(@(t,X)gradDirection(t,X,a),[0 inf],X0,options);
    [Qt,~] = CalcEigsDscd(Xf,n);
    [Qt2,~] = CalcEigsDscd(Lambda,n);
    Q = Qt2*Qt';
end

function status = check(~,X,flag,a)
    global Xf;
%     size(X)
%     size(a)
%     flag
    if (strcmp(flag,'done'))
        return;
    end
    a = diag(a);
    error  = 0.000001;
    n = length(X(:))^(1/2);
    X = reshape(X,n,n);
    residual = ((diag(diag(X))-a)./a)>error;
    if (sum(residual(:))==0)
        %it's time
        Xf = X;
        status = 1;
        return;
    end
    %fprintf('%d\n',t);
    status = 0;
end

function dX = gradDirection(~,X,a)
    %fprintf('%d\n',t);
    n = length(X(:))^(1/2);
    X = reshape(X,n,n);
    alphaX = diag(diag(X))-diag(a);
    t = alphaX*X-X*alphaX;
    dX = X*t-t*X;
    dX = reshape(dX,length(dX(:)),1);
end