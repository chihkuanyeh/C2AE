function [X2, XV2] = normalize(X2i, XV2i)
    if sum(sum(X2i == 0)) == 0
        X2 = (X2i + 1) ./ 2;
        XV2 = (XV2i + 1) ./ 2;
    else
        p = sum(X2i == 1, 2); n = sum(X2i == -1, 2);
        X2 = bsxfun(@rdivide, X2i == 1, p) + bsxfun(@rdivide, -(X2i == -1), n);
        X2(isnan(X2)) = 0;
        p = sum(XV2i == 1, 2); n = sum(XV2i == -1, 2);
        XV2 = bsxfun(@rdivide, XV2i == 1, p) + bsxfun(@rdivide, -(XV2i == -1), n);
        XV2(isnan(XV2)) = 0;
    end
end