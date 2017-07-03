clear
addpath ./util
% some parameters and dataset
datasets = {'mirflickr'};
reduction_rate = 0.8;
alpha = 2.0;
missing_rate = 0;
eta0 = 1e-4;
maxepoch = 50;
hiddentype = 'leaky_relu';
decay = 0.98;
momentum = 0.99;
l2penalty = 1e-3;

for z = 1:length(datasets)
    dataset = datasets{z};
    load(dataset);
    X2a = X2;
    XV2a = XV2;
    addpath ./deepnet/
    batchsize = 500;
    
    for i = 1:length(reduction_rate)
        K = round(size(X2, 2) * reduction_rate(i));
        NN1 = [512 512 K];
        NN2 = [512 K];
        for j = 1:length(alpha)
            for k = 1:length(missing_rate)
                [X2, XV2, ~, ~] = missing(X2a, XV2a, missing_rate(k));
                [X2, XV2] = normalize(X2, XV2);
                [F1opt,F2opt,F3opt,F4opt]=DCCAtrain_SGD(X1,X2,XV1,XV2,[],[],K,hiddentype,NN1,NN2,0,0,l2penalty,batchsize,eta0,alpha(j),decay,momentum,maxepoch,0);
                Fopt=[F1opt,F4opt];
                save_result(round(deepnetfwd(XTe1,Fopt)),XTe2,dataset,reduction_rate(i),alpha(j),missing_rate(k));
            end
        end
    end
end
