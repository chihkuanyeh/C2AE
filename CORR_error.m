function [E, G1, G2] = CORR_error(X1, X2)
    lambda = 0.5;
    [N, D] = size(X1);
    E = sum(sum((X1-X2).^2)) + lambda * (sum(sum((X1.'*X1-eye(D)).^2)) + sum(sum((X2.'*X2-eye(D)).^2)));
    G1 = 2*(X1-X2) + lambda * 4*X1*(X1.'*X1-eye(D));
    G2 = 2*(X2-X1) + lambda * 4*X2*(X2.'*X2-eye(D));
end
