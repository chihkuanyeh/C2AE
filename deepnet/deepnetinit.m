function F_init = deepnetinit(Layersizes,Layertypes,Decays)
% F_init=deepnetinit(Layersizes,Layertypes,Decays) initialize the weight 
%   parameters at each layer with random values.
%
% Inputs
%   Layersizes: vector containing number of units at each layer.
%   Layertypes: type of activations at each layer. Possible types include
%     'linear', 'sigmoid', 'tanh', 'relu', 'cubic', 'logistic', 'softmax'.
%   Decays: vector of weight decay parameters (l2 regularization) of 
%     weights at each layer.
%
% Outputs
%   F_init: cell array that contains all layers of the network. Each layer
%     has a field 'type' indicating the type of hidden activation, a field 
%     'units' indicating the output dimension of the layer, a filed 'l' 
%     indicating the weight decay parameter, and a field 'W' containing the
%     weight matrix.

Nlayers = length(Layersizes)-1;

if ~exist('Decays','var') || isempty(Decays)
  Decays = zeros(1,Nlayers);
end

if length(Layertypes)~=Nlayers
  error('Layertypes has a length not consistent with Layersizes!');
end

if length(Decays)~=Nlayers
  error('Weight decay parameters has a length not consistent with Layersizes!');
end

F_init = cell(1,Nlayers);
for j=1:Nlayers
  layer.type = Layertypes{j};
  layer.l = Decays(j);
  fan_in = Layersizes(j);
  fan_out = Layersizes(j+1);
  layer.units = fan_out;
  layer.W = zeros(fan_in+1,fan_out);
  switch layer.type
    case 'tanh'     % Suggested by Yoshua Bengio, normalized initialization.
      layer.W(1:end-1,:) = 2*(rand(fan_in,fan_out)-0.5)*sqrt(6)/sqrt(fan_in+fan_out);
    case 'cubic'     % Suggested by Yoshua Bengio, normalized initialization.
      layer.W(1:end-1,:) = 2*(rand(fan_in,fan_out)-0.5)*sqrt(6)/sqrt(fan_in+fan_out);
    case 'relu'
      layer.W(1:end-1,:) = 2*(rand(fan_in,fan_out)-0.5)*0.01; % sqrt(6)/sqrt(fan_in+fan_out);
      % Give some small postive bias so that initial activation is nonzero.
      layer.W(end,:) = rand(1,fan_out)*0.1;
    case 'leaky_relu'
      layer.W(1:end-1,:) = 2*(rand(fan_in,fan_out)-0.5)*0.01; % sqrt(6)/sqrt(fan_in+fan_out);
      % Give some small postive bias so that initial activation is nonzero.
      layer.W(end,:) = rand(1,fan_out)*0.1;
    case 'sigmoid'  % Suggested by Yoshua Bengio, 4 times bigger than tanh.
      layer.W(1:end-1,:) = 8*(rand(fan_in,fan_out)-0.5)*sqrt(6)/sqrt(fan_in+fan_out);
    case 'sigmoid_zero_mean'  % Suggested by Yoshua Bengio, 4 times bigger than tanh.
      layer.W(1:end-1,:) = 8*(rand(fan_in,fan_out)-0.5)*sqrt(6)/sqrt(fan_in+fan_out);
    otherwise       % The 1/sqrt(fan_in) rule, small random values.
      layer.W(1:end-1,:) = 2*(rand(fan_in,fan_out)-0.5)/sqrt(fan_in);
  end
  F_init{j} = layer;
end
