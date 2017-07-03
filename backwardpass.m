function [dE,delta]=backwardpass(delta,XX,F,dropprob)

% ****** ERROR BACK PROPAGATION ******
Nlayers=length(F);
if Nlayers==0
  dE=[];
else
  % XX is of length Nlayers+1.
  % Last layer, must be non-convolutional.
  j=Nlayers; N=size(delta,1);
  % To take into account the scale issue.
  delta=delta/(1-dropprob(j+1)); T=XX{j+1}*(1-dropprob(j+1)); delta(T==0)=0;
  switch lower(F{j}.type)
    case 'linear',
      % Do nothing.
    case 'relu',
      delta(T<=0)=0;
    case 'leaky_relu',
      delta(T<=0)=delta(T<=0)*0.1;
    case 'cubic',
      delta=delta./(1+T.^2);
    case 'sigmoid',
      delta=delta.*T.*(1-T);
    case 'sigmoid_zero_mean',
      delta=delta.*(0.5+T).*(0.5-T);
    case 'tanh',
      delta=delta.*(1-T.^2);
    case 'logistic',
      delta=delta.*T.*(1-T);
    case 'softmax',
      delta=delta.*T - repmat(sum(delta.*T,2),1,size(T,2)).*T;
    otherwise,
      error('Invalid layer type: %s\n',F{j}.type);
  end
  if Nlayers>1 && strcmpi(F{Nlayers-1}.type,'conv');
    de=[XX{j}.respfull ones(N,1)]'*delta;
  else
    de=[XX{j} ones(N,1)]'*delta;
  end
  de(1:end-1,:)=de(1:end-1,:) + 2*F{j}.l*F{j}.W(1:end-1,:);
  dE=de(:);
  % Prepare the delta for next layer.
  delta=delta * F{j}.W(1:end-1,:)';
  
  % Other layers.
  for j=Nlayers-1:-1:1
    T=XX{j+1};
    if strcmp(F{j}.type,'conv')
      % Reshape the deltas to the size of output feature maps. We start from here.
      delta=reshape(delta,N,F{j}.sizeout1,F{j}.sizeout2,F{j}.numoutputmaps);
      % Stretch. Inverse the pooling process.
      delta=delta(:,repmat(1:F{j}.sizeout1,F{j}.rowpoolratio,1),...
        repmat(1:F{j}.sizeout2,F{j}.colpoolratio,1),:);
      delta=delta(:,1:F{j}.sizeout_prepool1,1:F{j}.sizeout_prepool2,:);
      delta=delta.*T.respidx;
      switch F{j}.pooling
        case 'average',
          delta=delta./(F{j}.rowpoolratio*F{j}.colpoolratio);
      end
      % Inverse the nonlinearity.
      switch F{j}.sigmoid
        case 'sigmoid',
          delta=delta.*(T.respfull_prepool).*(1-T.respfull_prepool);
        case 'sigmoid_zero_mean',
          delta=delta.*(0.5+T.respfull_prepool).*(0.5-T.respfull_prepool);
        case 'tanh',
          delta=delta.*(1-(T.respfull_prepool).^2);
        case 'relu',
          delta(T.respfull_prepool==0)=0; % j, length(find(T.respfull_prepool==0))
        case 'leaky_relu',
          delta(T.respfull_prepool==0)=delta(T.respfull_prepool==0)*0.1; % j, length(find(T.respfull_prepool==0))
      end
      % delta is now of size [N, sizeout_prepool1, sizeout_prepool2, numoutputmaps].
      dbias=sum(reshape(delta,N*F{j}.sizeout_prepool1*F{j}.sizeout_prepool2,F{j}.numoutputmaps),1);
      % Fetch the layer below and reshape it to the size of input feature maps.
      if (j>1) && strcmp(F{j-1}.type,'conv')  % There are still conv layers below.
        lowerlayeroutput=XX{j}.respfull;
      else % It is the first layer, lower layer output is the input.
        lowerlayeroutput=XX{j};
      end
      lowerlayeroutput=reshape(lowerlayeroutput,N,F{j}.inputnumrows,F{j}.inputnumcols,F{j}.numinputmaps);
      lowerlayeroutput=repmat(lowerlayeroutput,[1 1 1 1 F{j}.numoutputmaps]);
      % lowerlayeroutput is of size [N inputnumrows intputnumcols numinputmaps numoutputmaps]
      rcW=F{j}.W(1:end-1,:);
      rcW=reshape(rcW,[1, F{j}.filternumrows, F{j}.filternumcols, F{j}.numinputmaps, F{j}.numoutputmaps]);
      rfilter=repmat(rcW,[N, 1, 1, 1, 1]);
      % rfilter is of size [N filternumrow filternumcols numinputmaps numoutputmaps].
      de=zeros(1, F{j}.filternumrows*F{j}.filternumcols*F{j}.numinputmaps,...
        F{j}.numoutputmaps, F{j}.sizeout_prepool1*F{j}.sizeout_prepool2);
      % de is of size [1 filternumrow*filternumcols numinputmaps prepoolimagesize].
      % if (j>1) && strcmp(F{j-1}.type,'conv')  % There are still conv layers below.
      delta_lower=zeros(N,F{j}.inputnumrows,F{j}.inputnumcols,F{j}.numinputmaps);
      % end
      for ai=1:F{j}.sizeout_prepool1
        for aj=1:F{j}.sizeout_prepool2
          % for each pixel in the convolved image.
          acts=reshape(delta(:,ai,aj,:),[N,1,F{j}.numoutputmaps]);
          % find subimage that contribute to the convolution.
          rowstart=(ai-1)*F{j}.rowstride+1;
          rowend=(ai-1)*F{j}.rowstride+F{j}.filternumrows;
          colstart=(aj-1)*F{j}.colstride+1;
          colend=(aj-1)*F{j}.colstride+F{j}.filternumcols;
          inblock=lowerlayeroutput(:, rowstart:rowend, colstart:colend, :, :);
          inblock=bsxfun(@times,reshape(inblock,[N,...
            F{j}.filternumrows*F{j}.filternumcols*F{j}.numinputmaps, F{j}.numoutputmaps]),acts);
          de(:,:,:, (ai-1)*F{j}.sizeout_prepool2+aj)=sum(inblock,1);
          
          % if (j>1) && strcmp(F{j-1}.type,'conv')  % There are still conv layers below.
          delta_lower(:,rowstart:rowend,colstart:colend,:)=...
            delta_lower(:,rowstart:rowend,colstart:colend,:) + ...
            sum(bsxfun(@times,rfilter,reshape(acts,N,1,1,1,F{j}.numoutputmaps)), 5);
          % end
        end
      end
      de=reshape(sum(de,4),F{j}.filternumrows*F{j}.filternumcols*F{j}.numinputmaps,F{j}.numoutputmaps);
      de=[de; dbias];
      de(1:end-1,:)=de(1:end-1,:) + 2*F{j}.l*F{j}.W(1:end-1,:);
      dE=[de(:); dE];
      % Prepare the delta for next layer.
      delta=delta_lower;
      if j==1 delta=reshape(delta,N,F{j}.inputnumrows*F{j}.inputnumcols*F{j}.numinputmaps); end
    else
      % Non-convolutional layers consider drop out.
      delta=delta/(1-dropprob(j+1)); T=T*(1-dropprob(j+1)); delta(T==0)=0;
      switch lower(F{j}.type)
        case 'linear',
          % Do nothing.
        case 'relu',
          delta(T<=0)=0;
        case 'leaky_relu',
          delta(T<=0)=delta(T<=0)*0.1;
        case 'cubic',
          delta=delta./(1+T.^2);
        case 'sigmoid',
          delta=delta.*T.*(1-T);
        case 'sigmoid_zero_mean',
          delta=delta.*(0.5+T).*(0.5-T);
        case 'tanh',
          delta=delta.*(1-T.^2);
        case 'logistic',
          delta=delta.*T.*(1-T);
        otherwise,
          error('Invalid layer type: %s\n',F{j}.type);
      end
      if j>1 && strcmpi(F{j-1}.type,'conv');
        de=[XX{j}.respfull ones(N,1)]'*delta;
      else
        de=[XX{j} ones(N,1)]'*delta;
      end
      de(1:end-1,:)=de(1:end-1,:) + 2*F{j}.l*F{j}.W(1:end-1,:);
      dE=[de(:); dE];
      % Prepare the delta for next layer.
      delta=delta * F{j}.W(1:end-1,:)';
    end
  end
  
  % ****** END OF ERROR BACKPROPAGATION ******
end