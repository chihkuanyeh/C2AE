function [F1opt,F2opt,F3opt,F4opt]=DCCAtrain_SGD( ...
  X1,X2,XV1,XV2,XTe1,XTe2,K,hiddentype,NN1,NN2,rcov1,rcov2,l2penalty, ...
  batchsize,eta0,alpha,decay,momentum,maxepoch,savefile,randseed)

if ~exist('savefile','var')
    savefile = 1;
end
if ~exist('randseed','var') || isempty(randseed)
  randseed=1;
end
rng(randseed);

%% Filename to save intermediate result.
filename=['result_K=' num2str(K) ...
  '_rcov1=' num2str(rcov1) '_rcov2=' num2str(rcov2) ...
  '_l2penalty=' num2str(l2penalty) ...
  '_batchsize=' num2str(batchsize) ...
  '_eta0=' num2str(eta0) '_decay=' num2str(decay) ...
  '_momentum=' num2str(momentum) ...
  '_maxepoch=' num2str(maxepoch) ...
  '.mat'];

if exist(filename,'file')
  
  load(filename,'randseed','F1opt','F2opt','F3opt','F4opt','F1','F2','F3','F4','TIME',...
    'eta','delta','optvalid','rrr');
  its=length(CORR_train)-1;
  if its>=maxepoch;
    fprintf('Neural networks have already been trained!\nExiting ...\n');
    return;
  else
    fprintf('Neural networks trained halfway!\nLoading ...\n');
    [N,D1]=size(X1);  [~,D2]=size(X2);
  end
else
  
  % fprintf('Result will be saved in %s\n',filename);
  [N,D1]=size(X1); [~,D2]=size(X2);
  %% Set view1 architecture.
  Layersizes1=[D1 NN1];  Layertypes1={};
  for nn1=1:length(NN1)-1;
    Layertypes1=[Layertypes1, {hiddentype}];
  end
  % last layer is sigmoid shifted to zero mean.
  Layertypes1{end+1}='sigmoid_zero_mean';
  %% Set view2 architecture.
  Layersizes2=[D2 NN2];  Layertypes2={};
  for nn2=1:length(NN2)-1;
    Layertypes2=[Layertypes2, {hiddentype}];
  end
  Layertypes2{end+1}='sigmoid_zero_mean';
    %% Set view3 architecture.
  Layersizes3=fliplr(Layersizes1);  Layertypes3={};
  for nn1=1:length(NN1)-1;
    Layertypes3=[Layertypes3, {hiddentype}];
  end
  Layertypes3{end+1}='sigmoid';
  %% Set view4 architecture.
  Layersizes4=fliplr(Layersizes2);  Layertypes4={};
  for nn2=1:length(NN2)-1;
    Layertypes4=[Layertypes4, {hiddentype}];
  end
  Layertypes4{end+1}='sigmoid';
  %% Random initialization of weights.
  F1=deepnetinit(Layersizes1,Layertypes1);
  F2=deepnetinit(Layersizes2,Layertypes2);
  F3=deepnetinit(Layersizes3,Layertypes3);
  F4=deepnetinit(Layersizes4,Layertypes4);
  % we only have 3 network component for C2AE
  F3=F4;
  
  %% L2 penalty on weights is used for DCCA training.
  for j=1:length(F1)  F1{j}.l=l2penalty;  end
  for j=1:length(F2)  F2{j}.l=l2penalty;  end
  for j=1:length(F3)  F3{j}.l=l2penalty;  end
  for j=1:length(F4)  F4{j}.l=l2penalty;  end
  
  %% Compute canonical correlations at the outputs.
 

  its=0; TIME=0; delta=0; eta=eta0; rrr=[];
  optvalid=0; F1opt=F1; F2opt=F2; F3opt=F3; F4opt=F4;
  if savefile
    save(filename,'randseed','F1opt','F2opt','F3opt','F4opt','F1','F2','F3','F4','TIME',...
      'eta','delta','optvalid','rrr');
  end
end

%% Concatenate the weights in a long vector.
VV=[];
Nlayers=length(F1); net1=cell(1,Nlayers);
for k=1:Nlayers
  VV=[VV; F1{k}.W(:)];  net1{k}=rmfield(F1{k},'W');
end
Nlayers=length(F2); net2=cell(1,Nlayers);
for k=1:Nlayers
  VV=[VV; F2{k}.W(:)];  net2{k}=rmfield(F2{k},'W');
end
Nlayers=length(F3); net3=cell(1,Nlayers);
for k=1:Nlayers
  VV=[VV; F3{k}.W(:)];  net3{k}=rmfield(F3{k},'W');
end
Nlayers=length(F4); net4=cell(1,Nlayers);
for k=1:Nlayers
  VV=[VV; F4{k}.W(:)];  net4{k}=rmfield(F4{k},'W');
end
fprintf('Number of weight parameters: %d\n',length(VV));

%% Use GPU if equipped. GPU significantly speeds up optimization.
if gpuDeviceCount>0
  fprintf('GPU detected. Trying to use it ...\n');
  try
    VV=gpuArray(VV);
    X1=gpuArray(X1);
    X2=gpuArray(X2);
    fprintf('Using GPU ...\n');
  catch
  end
end

Xn = bsxfun(@rdivide, X2, sqrt(sum(X2 > 0, 1)));
Xn0 = bsxfun(@rdivide, (1-X2), sqrt(sum(X2 == 0, 1)));
W = gather((Xn.' * (Xn0) + (Xn0).' * (Xn))/2);
W = W./(mean(mean(W)));

%% Start stochastic gradient descent.
numbatches=ceil(N/batchsize);
while its<maxepoch
    eta=eta0*decay^its; % Reduce learning rate.
  t0=tic;
  rp=randperm(N);   % Shuffle the data set.
  for i=1:numbatches
    idx1=(i-1)*batchsize+1;
    idx2=min(i*batchsize,N);
    idx=[rp(idx1:idx2),rp(1:max(0,i*batchsize-N))];
    X1batch=X1(idx,:);  X2batch=X2(idx,:);
    
    % Evaluate stochastic gradient.
    [E,grad]=DCCA_grad(VV,X1batch,X2batch,net1,net2,net3,net4,W,K,alpha,rcov1,rcov2);
    if isempty(rrr), rrr=grad; end
    rrr=sqrt((rrr.^2)*0.9+(grad.^2)*0.1);
    grad=grad./rrr;
    grad(isnan(grad))=0;
    delta=momentum*delta-eta*grad;  % Momentum.
    VV=VV + delta;
  end
  
  %% Record the time spent for each epoch.
  its=its+1; TIME=[TIME, toc(t0)];
  
  %% Use GPU if equipped. GPU significantly speeds up optimization.
if gpuDeviceCount>0
  VV=gather(VV);
  X1=gpuArray(X1);
  X2=gpuArray(X2);
  rrr = gather(rrr);
end

  %% Assemble the networks.
  idx=0;
  D=size(X1,2);
  for j=1:length(F1)
    if strcmp(F1{j}.type,'conv')
      convdin=F1{j}.filternumrows*F1{j}.filternumcols*F1{j}.numinputmaps;
      convdout=F1{j}.numoutputmaps;
      W_seg=VV(idx+1:idx+(convdin+1)*convdout);
      F1{j}.W=reshape(W_seg,convdin+1,convdout);
      idx=idx+(convdin+1)*convdout;
      D=F1{j}.units;
    else
      units=F1{j}.units;
      W_seg=VV(idx+1:idx+(D+1)*units);
      F1{j}.W=reshape(W_seg,D+1,units);
      idx=idx+(D+1)*units; D=units;
    end
  end
  
  D=size(X2,2);
  for j=1:length(F2)
    if strcmp(F2{j}.type,'conv')
      convdin=F2{j}.filternumrows*F2{j}.filternumcols*F2{j}.numinputmaps;
      convdout=F2{j}.numoutputmaps;
      W_seg=VV(idx+1:idx+(convdin+1)*convdout);
      F2{j}.W=reshape(W_seg,convdin+1,convdout);
      idx=idx+(convdin+1)*convdout;
      D=F2{j}.units;
    else
      units=F2{j}.units;
      W_seg=VV(idx+1:idx+(D+1)*units);
      F2{j}.W=reshape(W_seg,D+1,units);
      idx=idx+(D+1)*units; D=units;
    end
  end
  
  D=K;
  for j=1:length(F3)
    units=F3{j}.units;
    W_seg=VV(idx+1:idx+(D+1)*units);
    F3{j}.W=reshape(W_seg,D+1,units);
    idx=idx+(D+1)*units; D=units;
  end
  
  D=K;
  for j=1:length(F4)
    units=F4{j}.units;
    W_seg=VV(idx+1:idx+(D+1)*units);
    F4{j}.W=reshape(W_seg,D+1,units);
    idx=idx+(D+1)*units; D=units;
  end
  
  %% Compute correlations and errors.

  X_tune=deepnetfwd(XV1,F1);
  PP = deepnetfwd(XV1,[F1,F4]);
  [EE1, ~] = BR_error(PP, XV2, W);
  [micro_f1, macro_f1] = f1_score(round(PP), XV2);
  PP = deepnetfwd(XV2,[F2,F4]);
  [EE2, ~] = BR_error(PP, XV2, W);
  [micro_f2, macro_f2] = f1_score(round(PP), XV2);
  
  fprintf('Epoch %d: ', its);
  fprintf('err = %f, micro_f1 = %f, macro_f1 = %f\n', EE1, micro_f1, macro_f1);
  if its<10, fprintf('         '); else fprintf('          '); end
  fprintf('err = %f, micro_f1 = %f, macro_f1 = %f ', EE2, micro_f2, macro_f2);
  score = micro_f1 + macro_f1;
  % save best validation to Fopt for the average of micro_f1 and macro_f1
  if score>optvalid
    optvalid=score;
    fprintf('getting better score\n');
    F1opt=F1;  F2opt=F2;  F3opt=F3;  F4opt=F4;
  else
    fprintf('getting worse score\n');
  end
  if savefile
    save(filename,'randseed','F1opt','F2opt','F3opt','F4opt','F1','F2','F3','F4','TIME', ...
      'eta','delta','optvalid','rrr');
  end
end
