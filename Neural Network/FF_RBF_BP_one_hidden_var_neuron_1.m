%{
    * Creator:          Chuong Nguyen
    * Email:            chuong19111996@gmail.com    
                        nguyenhoangkhanhchuong1996@gmail.com
    * File:             A simple FFNN with RBF
    * Description:      A simple FFNN with RBF + BP 
                        The structure of the network is shown below
                        One feature input
                        One hidden layer with user defined number 
                        One output at the output layer
                        
                        If this is too cumbersome to read/understand, it is
                        advised to go back and read FF_BP_simple_1.m 
%}

%{
Network Structure:
                           __ h1 __
                          _w1    w(n+1)_
                         _          _
                        x__w2_h2 _w(n+2)__y
                         _          _ 
                          _wn    w(n+n)_
                           __ hn __

%}

%{
    * Key parts:
        (1) Import Data & Separate Dataset
        (2) Parameters
        (3) Derivative of Cost function (w.r.t. each param) for Training Process
        (5) Train Data with Back-propagation/Gradient Descent
        (6) Test Model(s)
%}

%{
    * Experience:
        (1) One mistake was made, this is a linear regression problem, so
        we apply linear activation function at output. I applied sigmoid
        unknowningly for output layer so the output is always flat 1.

    * References:
        https://stats.stackexchange.com/questions/209646/delimitation-feed-forward-and-radial-basis-networks
        
        MLP: uses dot products (between inputs and weights) and sigmoidal 
        activation functions (or other monotonic functions) and training 
        is usually done through backpropagation for all layers (which can 
        be as many as you want);

        RBF: uses Euclidean distances (between inputs and weights) and 
        Gaussian activation functions, which makes neurons more locally 
        sensitive. Also, RBFs may use backpropagation for learning, or 
        hybrid approaches with unsupervised learning in the hidden layer 
        (they have just 1 hidden layer).

%}

%% Clean
clc
clear

%% Import Data & Separate Dataset
% Read data
data = readmatrix('dataset1.xlsx');
x_data = data(:,1);
y_data = data(:,2);

% Separate data
ratio = 0.7;                            % Define separate ratio
datapt_num = length(data(:,1));         % Number of data points
trainpt_num = round(datapt_num*ratio);  % Rounded number of training points

% Training set
x_train = x_data(1:trainpt_num);        % We want x_data*ratio
y_train = y_data(1:trainpt_num);        % We want y_data*ratio
% figure
% plot(x_train,y_train);

% Validation set
x_valid = x_data(trainpt_num+1:end);    % We want x_data*(1-ratio)
y_valid = y_data(trainpt_num+1:end);    % We want y_data*(1-ratio)
% figure
% plot(x_valid,y_valid);

% Normalize Data

%% Parameters

% Ask user for desired number of neurons
prompt = "Number of neurons? ";
num_neuron = input(prompt);
w_init = zeros(2*num_neuron,1);              % For 1 feature input and 1 output
b_init = zeros(2,1);                         % For 1 feature input and 1 output

% Ask user whether they want to manually input the weights & biases
prompt = "Would you like to manually input the weight and bias? 1 is 'yes', 0 is 'no' ";
manual_bool = input(prompt);

if manual_bool == 1
    % The case where user want to manually input
    for i = 1:length(w_init)
        prompt = strcat("Weight ",string(i)," is ");
        w_init(i) = input(prompt);
    end
    for i = 1:length(b_init)
        prompt = strcat("Bias ",string(i)," is ");
        b_init(i) = input(prompt);
    end
else
    % The case where user doesn't want to manually input
    % Use default values for weights & biases
    w_default = 0.1;
    for i = 1:length(w_init)
        w_init(i) = w_default;
    end
    b_default = 0.01;
    for i = 1:length(b_init)
        b_init(i) = b_default;
    end
    prompt = strcat("Default weight & bias is set to ",string(w_default), " & ", string(b_default))
end

% Add weight & bias to initial parameters, this step can be grouped with
% above steps but is kept separated for clarity.

% Arrange weights & biases as initial values of parameter array

% The length of param_init is equal to sum length of both weights & biases
param_init = zeros(length(w_init) + length(b_init),1);
% Assign weights
for i = 1:length(w_init)
    param_init(i) = w_init(i);
end
% Assign biases
for i = 1:length(b_init)
    % param_init picks up from last element of previous loop
    param_init(length(w_init)+i) = b_init(i);
end
    
%% Back-propagation

% Pre-calculation of derivative of gradient descent for all weight.

% Define symbolic variables
syms x y out                        % feature input, measured output, output
w = sym('w',[1 length(w_init)]);    % symbolic weight
b = sym('b',[1 length(b_init)]);    % symbolic bias
zh = sym('zh',[1 num_neuron]);      % symbolic input func to hidden layer
h = sym('h',[1 num_neuron]);        % symbolic activation func

% Input to hidden layer
for i = 1:num_neuron
    % zh(i) = w(i)*x + b(1);            % of original FFNN
    zh(i) = sqrt(w(i)-x)^2 + b(1);      % of RBF    
    %  Activation func at hidden layer
    % h(i) = sigmoid_act_func(zh(i));   % of original FFNN
    h(i) = gaussian_act_func(zh(i));    % of RBF 
end

% Hidden layer to output layer
zo1 = w(num_neuron+1)*h(1);
for i = 2:num_neuron
    zo1 = zo1 + w(num_neuron+i)*h(i);
end
zo1 = zo1 + b(2);
% Activation func at output layer
out = zo1*1.0;      % Use linear activation.

% Cost function
err = (1/2)*(y-out)^2;

% Partial Derivative of Cost function w.r.t. all weights & biases
err_diff = sym('err_diff', [1 length(param_init)]);
% Partial Derivative of Cost function w.r.t. all weights
for i = 1:length(w)
    err_diff(i) = diff(err,w(i));
end
% Partial Derivative of Cost function w.r.t. all biases
for i = 1:length(b)
    err_diff(length(w)+i) = diff(err,b(i));
end

%% Training Process
iteration_num = 1000;
learning_rate = 0.01;

param_old = param_init;
% We want param_new to have length sum of weights & biases, so just use a
% variable that have that, which is param_old in this case
param_new = zeros(length(param_old),1); 

for k = 1:iteration_num
    
    % Separate param_old into weights & biases to substitute to symbolic
    % function. Also transpose them to match the array form.
    w_param = param_old(1:length(w)); w_param = w_param';
    b_param = param_old(length(w)+1:end); b_param = b_param';
    
    for i = 1:length(param_init)
        err_de = 0;
        for j = 1:length(x_train)
            err_de = err_de + subs(err_diff(i),[x,y,w,b],[x_train(j),y_train(j),w_param,b_param]);
        end
        err_de = double(err_de)/length(x_train);
        param_new(i) = param_old(i) - learning_rate*err_de;
    end
    param_old = param_new;
end

%% Test Model(s)
y_new = zeros(length(y_data),1);

% Separate param_old into weights & biases to substitute to symbolic
% function. Also transpose them to match the array form.
w_param = param_new(1:length(w)); w_param = w_param';
b_param = param_new(length(w)+1:end); b_param = b_param';
for i = 1:length(x_data)
    y_new(i) = subs(out,[x,w,b],[x_data(i),w_param,b_param]);
end
figure
hold on
plot(x_data,y_data)
plot(x_data,y_new)
hold off

error = zeros(length(y_data),1);
for i = 1:length(x_data)
    error(i) = y_data(i) - y_new(i);
end
figure
plot(error)