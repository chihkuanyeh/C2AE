function [E,grad]=DCCA_grad(VV,X1,X2,F1,F2,F3,F4,W,K,alpha,rcov1,rcov2,dropprob1,dropprob2)

if ~exist('dropprob1','var') || isempty(dropprob1)
  dropprob1=[0 0*ones(1,length(F1))];
end
if ~exist('dropprob2','var') || isempty(dropprob2)
  dropprob2=[0 0*ones(1,length(F2))];
end

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
    convdin=F2{j}.filternumrows*F2{j}.filternumcols*F2{j}.F2{j}.numinputmaps;
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

[XX1,R1]=forwardpass(X1,0,F1,dropprob1);
[XX2,R2]=forwardpass(X2,0,F2,dropprob2);
[XX3,R3]=forwardpass(XX1{end},R1,F3,dropprob1);
[XX4,R4]=forwardpass(XX2{end},R2,F4,dropprob2);
% Compute objective function and derivative w.r.t. last layer output.
[~,G3]=BR_error(XX3{end},X2,W);
[~,G4]=BR_error(XX4{end},X2,W);
[grad3,de3]=backwardpass(G3 / alpha,XX3,F3,dropprob1);
[grad4,de4]=backwardpass(G4 / alpha,XX4,F4,dropprob2);
[~, G1, G2] = CORR_error(XX1{end},XX2{end});
% Note that the loss in the output layer is only propogated to F2 and not
% F1, that is only to Fe but not Fx
E = 0; G1 = G1; G2 = G2 + de4;

[grad1,~]=backwardpass(G1,XX1,F1,dropprob1);
[grad2,~]=backwardpass(G2,XX2,F2,dropprob2);
grad=[grad1(:); grad2(:); grad3(:); grad4(:)];
