function [X2m, XV2m, X2, XV2] = missing(X2, XV2, rate)
    X2c = mat2cell(X2, ones(size(X2, 1), 1), size(X2, 2));
    X2mc = cellfun(@(a) F(a, rate), X2c, 'UniformOutput', false);
    X2m = cell2mat(X2mc);
    
    XV2c = mat2cell(XV2, ones(size(XV2, 1), 1), size(XV2, 2));
    XV2mc = cellfun(@(a) F(a, rate), XV2c, 'UniformOutput', false);
    XV2m = cell2mat(XV2mc);
end

function xm = F(x, rate)
    n = length(x);
    n2 = round(n * rate);
    r = randperm(n);
    xm = x * 2 - 1;
    xm(r(1:n2)) = 0;
    if sum(xm > 0) == 0 && sum(x) > 0
        i = find(x);
        ii = randi(length(i));
        xm(i(ii)) = 1;
        xm(r(n2 + 1)) = 0;
    end
end