function [XX,R]=forwardpass(X,r,F,dropprob)
% Compute all intermediate output and regularization.

N=size(X,1);
Nlayers=length(F);
R=r;
T=X;
XX=cell(1,Nlayers+1);

% Drop out the inputs.
% j=1; tmp=rand(size(T)); T(tmp<dropprob(j))=0; T=T/(1-dropprob(j)); XX{j}=T;
XX{1}=T;
  
% ****** FEED FORWARD ******
for j=1:Nlayers
  
  if strcmp(F{j}.type,'conv')
    % Reshape weights.
    R=R+F{j}.l*sum(sum(F{j}.W(1:end-1,:).^2));
    % Reshape input.
    T=reshape(T,N,F{j}.inputnumrows,F{j}.inputnumcols,F{j}.numinputmaps);
    OUT.respfull_prepool=zeros(N,F{j}.sizeout_prepool1,F{j}.sizeout_prepool2,F{j}.numoutputmaps);
    OUT.respidx=zeros(N,F{j}.sizeout_prepool1,F{j}.sizeout_prepool2,F{j}.numoutputmaps);
    OUT.respfull=zeros(N,F{j}.sizeout1,F{j}.sizeout2,F{j}.numoutputmaps);
    for filteridx=1:F{j}.numoutputmaps
      % Reshape filters, there is one filter for every output feature map.
      Wconv=reshape(F{j}.W(1:end-1,filteridx),1,F{j}.filternumrows,F{j}.filternumcols,F{j}.numinputmaps);
      bconv=F{j}.W(end,filteridx);
      Wconv=Wconv(1,end:-1:1,end:-1:1,end:-1:1);
      % Compute filter response.
      resp=convn(T,Wconv,'valid');
      % Use strides.
      resp=resp(:,1:F{j}.rowstride:end,:);
      resp=resp(:,:,1:F{j}.colstride:end);
      resp=resp+bconv;
      switch F{j}.sigmoid
        case 'sigmoid',
          resp=1./(1+exp(-resp));
        case 'sigmoid_zero_mean',
          resp=1./(1+exp(-resp))-0.5;
        case 'tanh',
          expa=exp(resp); expb=exp(-resp);
          resp=(expa - expb) ./ (expa + expb);
        case 'relu',
          resp(resp<0)=0;
        case 'leaky_relu',
          resp(resp<0)=resp(resp<0)*0.1;
      end
      OUT.respfull_prepool(:,:,:,filteridx)=resp;
      % Start pooling.
      % resp is of dimension [N, sizeout_prepool1, sizeout_prepool2, 1].
      switch F{j}.pooling
        case 'max',
          [resp, respidx]=maxpool(resp, [F{j}.rowpoolratio F{j}.colpoolratio]);
        case 'average',
          [resp, respidx]=avgpool(resp, [F{j}.rowpoolratio F{j}.colpoolratio]);
      end
      OUT.respidx(:,:,:,filteridx)=respidx;
      OUT.respfull(:,:,:,filteridx)=resp;
    end
    OUT.respfull=reshape(OUT.respfull,N,F{j}.sizeout1*F{j}.sizeout2*F{j}.numoutputmaps);
    T=OUT.respfull; XX{j+1}=OUT;
  else
    R=R+F{j}.l*sum(sum(F{j}.W(1:end-1,:).^2)); % Regularization for the weights only.
    T=[T ones(N,1)]*F{j}.W;
    switch lower(F{j}.type)
      case 'linear',
        % Do nothing.
      case 'relu',
        T(T<0)=0;
      case 'leaky_relu',
        T(T<0)=T(T<0)*0.1;
      case 'cubic',
        T=nthroot(1.5*T+sqrt(2.25*T.^2+1),3)+nthroot(1.5*T-sqrt(2.25*T.^2+1),3);
        T=real(T);
      case 'sigmoid',
        T=1./(1+exp(-T));
      case 'sigmoid_zero_mean',
        T=1./(1+exp(-T))-0.5;
      case 'tanh',
        expa=exp(T); expb=exp(-T);
        T=(expa - expb) ./ (expa + expb);
      case 'logistic',
        if size(F{j}.W,2)~=1
          error('logistic is only used for binary classification\n');
        else
          T=1./(1+exp(-T));
        end
      case 'softmax',
        T=exp(T); s=sum(T,2); T=diag(sparse(1./s))*T;
      otherwise,
        error('Invalid layer type: %s\n',F{j}.type);
    end
    % Drop out non-convolutional layer.
    tmp=rand(size(T)); T(tmp<dropprob(j+1))=0; T=T/(1-dropprob(j+1)); XX{j+1}=T;
  end
end