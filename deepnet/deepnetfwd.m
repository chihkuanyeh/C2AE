% [T,e,F]=deepnetfwd(X,F,Y,W) feed-forwards a network and computes error at the output end.
%
% In:
%   X: input data, NxD matrix.
%   Y: target data, Nxd matrix.
%   F: the deep net.
%   W: a long vector containing weights and biases at all layers, if it is
%      specified, weights in F (the 'W' field, if there is any) will be replaced.
%
% Out:
%   T: output of the network.
%   e: error of the output, if Y is specified. Notice that the regularization error is not added.
%   F: (modified) F, with updated weights.

% Copyright (c) 2012 by Miguel A. Carreira-Perpinan and Weiran Wang.

function [T,e,F] = deepnetfwd(X,F,Y,W)

if length(F)==0
  T=X;
else
  if nargin>3
    D = size(X,2); idx = 0;
    for j = 1:length(F);
      units = F{j}.units;
      W_seg = W(idx+1:idx+(D+1)*units);
      F{j}.W = reshape(W_seg,D+1,units);
      idx = idx+(D+1)*units; D = units;
    end
  end
  
  N=size(X,1);  D=F{end}.units;  T=[];
  B=5000; % Block size.
  NUMBATCHES=ceil(N/B);
  
  for i=1:NUMBATCHES
    startidx=(i-1)*B+1;
    endidx=min(i*B,N);
    
    Tseg = deepnetfwd_chunk(X(startidx:endidx,:),F);
    T=[T; Tseg];
  end
  
  if (nargin>2) && (nargout>1)
    switch F{end}.type
      case 'linear',
        e = sum(sum((Y-T).^2));
      case 'relu',
        e = sum(sum((Y-T).^2));
      case 'leaky_relu',
        e = sum(sum((Y-T).^2));
      case 'cubic',
        e = sum(sum((Y-T).^2));
      case 'sigmoid',
        e = sum(sum((Y-T).^2));
      case 'sigmoid_zero_mean',
        e = sum(sum((Y-T).^2));
      case 'tanh',
        e = sum(sum((Y-T).^2));
      case 'logistic',
        e = sum(-Y.*log(T)-(1-Y).*log(1-T));
      case 'softmax',
        T(T==0)=eps;
        e = sum(sum(-Y.*log(T)));
      otherwise,
        error('Invalid layer type: %s\n',F{end}.type);
    end
    
  end
end


function [T,e,F]=deepnetfwd_chunk(X,F,Y,W)

Nlayers=length(F);

if nargin>3
  D=size(X,2); idx=0;
  for j=1:Nlayers
    if strcmp(F{j}.type,'conv')
      convdin=F{j}.filternumrows*F{j}.filternumcols*F{j}.F{j}.numinputmaps;
      convdout=F{j}.numoutputmaps;
      W_seg=W(idx+1:idx+(convdin+1)*convdout);
      F{j}.W=reshape(W_seg,convdin+1,convdout);
      idx=idx+(convdin+1)*convdout;
      D=F{j}.units;
    else
      units=F{j}.units;
      W_seg=W(idx+1:idx+(D+1)*units);
      F{j}.W=reshape(W_seg,D+1,units);
      idx=idx+(D+1)*units; D=units;
    end
  end
end

T=X; N=size(X,1);
for j=1:Nlayers
  % % %     fprintf('feeding forward, layer %d, type: %s...\n',j,F{j}.type);
  
  switch F{j}.type
    case 'linear',
      T=[T, ones(N,1)]*F{j}.W;
    case 'relu',
      T=[T, ones(N,1)]*F{j}.W;
      T(T<0)=0; % sum(sum(T==0))/numel(T), std(T(T>0))
    case 'leaky_relu',
      T=[T, ones(N,1)]*F{j}.W;
      T(T<0)=T(T<0)*0.1; % sum(sum(T==0))/numel(T), std(T(T>0))
    case 'cubic',
      T=[T, ones(N,1)]*F{j}.W;
      T=nthroot(1.5*T+sqrt(2.25*T.^2+1),3)+nthroot(1.5*T-sqrt(2.25*T.^2+1),3);
    case 'sigmoid',
      T=[T, ones(N,1)]*F{j}.W;
      T=1./(1+exp(-T));
    case 'sigmoid_zero_mean',
      T=[T, ones(N,1)]*F{j}.W;
      T=1./(1+exp(-T)) - 0.5;
    case 'tanh',
      T=[T, ones(N,1)]*F{j}.W;
      expa=exp(T); expb=exp(-T);
      T=(expa - expb) ./ (expa + expb);
    case 'logistic',
      T=[T, ones(N,1)]*F{j}.W;
      T=1./(1+exp(-T));
    case 'softmax',
      T=[T, ones(N,1)]*F{j}.W;
      T=exp(T); s=sum(T,2);
      T=diag(sparse(1./s))*T;
    case 'conv',
      layer=F{j};
      % Reshape input.
      T=reshape(T,N,layer.inputnumrows,layer.inputnumcols,layer.numinputmaps);
      respfull=zeros(N,layer.sizeout1,layer.sizeout2,layer.numoutputmaps);
      for filteridx=1:layer.numoutputmaps
        % Reshape filters, there is one filter for every output feature map.
        Wconv=reshape(layer.W(1:end-1,filteridx),1,layer.filternumrows,layer.filternumcols,layer.numinputmaps);
        bconv=layer.W(end,filteridx);
        % Compute filter response.
        Wconv=Wconv(1,end:-1:1,end:-1:1,end:-1:1);
        resp=convn(T,Wconv,'valid');
        % Use strides.
        resp=resp(:,1:layer.rowstride:end,:);
        resp=resp(:,:,1:layer.colstride:end);
        resp=resp+bconv;
        switch layer.sigmoid
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
        % Start pooling.
        % resp is of dimension [N, layer.sizeout_prepool1, layer.sizeout_prepool2, 1].
        switch layer.pooling
          case 'max',
            [resp, ~]=maxpool(resp, [layer.rowpoolratio layer.colpoolratio]);
          case 'average',
            [resp, ~]=avgpool(resp, [layer.rowpoolratio layer.colpoolratio]);
        end
        respfull(:,:,:,filteridx)=resp;
      end
      % Always flatten the responses.
      T=reshape(respfull,N,layer.sizeout1*layer.sizeout2*layer.numoutputmaps);
    otherwise,
      error('Invalid layer type: %s\n',F{j}.type);
  end
end

if (nargin>2) && (nargout>1)
  switch F{end}.type
    case 'linear',
      e=sum(sum((Y-T).^2));
    case 'relu',  % Unlikely to happen.
      e=sum(sum((Y-T).^2));
    case 'leaky_relu',  % Unlikely to happen.
      e=sum(sum((Y-T).^2));
    case 'cubic',  % Unlikely to happen.
      e=sum(sum((Y-T).^2));
    case 'sigmoid',
      e=sum(sum((Y-T).^2));
    case 'sigmoid_zero_mean',
      e=sum(sum((Y-T).^2));
    case 'tanh',
      e=sum(sum((Y-T).^2));
    case 'logistic',
      e=sum(-Y.*log(T)-(1-Y).*log(1-T));
    case 'softmax',
      T(T==0)=eps;
      e=sum(sum(-Y.*log(T)));
    otherwise,
      error('Invalid layer type: %s\n',F{end}.type);
  end
end
