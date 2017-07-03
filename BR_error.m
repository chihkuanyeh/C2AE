% loss function used in our paper
function [E, G] = BR_error(p, y, w)
    if gpuDeviceCount>0
        p = gather(p);
        y = gather(y);
    end
    if sum(sum(y < 0)) == 0
        y = y - 0.5;
    end
    [N, D] = size(y);
    pc = mat2cell(p, ones(1, N), D);
    yc = mat2cell(y, ones(1, N), D);
    [Ec, Gc] = cellfun(@(a, b) error(a, b, w), pc, yc, 'UniformOutput', false);
    E = mean(cell2mat(Ec));
    G = cell2mat(Gc);
    if gpuDeviceCount>0
        E = gpuArray(E);
        G = gpuArray(G);
    end
end

function [e, g] = error(p, y, w)
    YEE = (y > 0);
    LIN = (y < 0);
    weight = w(LIN, YEE);
    num = sum(YEE) * sum(LIN);
    p1 = p(YEE);
    p0 = p(LIN);
    err = bsxfun(@minus, p1, p0.');
    % 5 is a fixed parameter in our loss function, we fix the parameter 
    %througout all experiments
    err =  exp(-5 * err) ./ num;
    e = sum(sum(err));
    g = zeros(size(y));
    g(YEE) = -sum(err, 1);
    g(LIN) = sum(err, 2).';
end
