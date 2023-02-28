%{
    * Creator:          Chuong Nguyen
    * Email:            chuong19111996@gmail.com    
                        nguyenhoangkhanhchuong1996@gmail.com
    * File:             A simple FFNN
    * Description:      A simple FFNN with BP 
                        The structure of the network is shown below
                        One feature input
                        One hidden layer with three neuron
                        One output at the output layer
%}

%{
Network Structure:
                           __ h1 __
                          _w1    w4_
                         _          _
                        x__w2_h2 _w5__y
                         _          _ 
                          _w3    w6_
                           __ h3 __

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
x_train = x_data(1:trainpt_num);
y_train = y_data(1:trainpt_num);
% figure
% plot(x_train,y_train);

% Validation set
x_valid = x_data(trainpt_num+1:end);
y_valid = y_data(trainpt_num+1:end);
% figure
% plot(x_valid,y_valid);

% Normalize Data

%% Parameters
% Input to hidden layer
w1_val = 1;
w2_val = 1;
w3_val = 1;
b1_val = 1;
% Hidden layer to output layer
w4_val = 1;
w5_val = 1;
w6_val = 1;
b2_val = 1;
param_init = [w1_val,w2_val,w3_val,w4_val,w5_val,w6_val,b1_val,b2_val];
%% Back-propagation

%{
Network Structure:
                           __ h1 __
                          _w1    w4_
                         _          _
                        x__w2_h2 _w5__y
                         _          _ 
                          _w3    w6_
                           __ h3 __

%}

% Pre-calculation of derivative of gradient descent for all weight.
syms x y out w1 w2 w3 w4 w5 w6 b1 b2

% Input to hidden layer
zh1 = w1*x + b1;
zh2 = w2*x + b1;
zh3 = w3*x + b1;
% Activation func at hidden layer
h1 = sigmoid_act_func(zh1);
h2 = sigmoid_act_func(zh2);
h3 = sigmoid_act_func(zh3);
% Hidden layer to output layer
zo1 = w4*h1 + w5*h2 + w6*h3 + b2;
% Activation func at output layer
out = zo1*1.0;

% Cost function
err = (1/2)*(y-out)^2;

% Partial Derivative of cost function w.r.t. all weight and bias
err_w1_diff = diff(err,w1);
err_w2_diff = diff(err,w2);
err_w3_diff = diff(err,w3);
err_w4_diff = diff(err,w4);
err_w5_diff = diff(err,w5);
err_w6_diff = diff(err,w6);
err_b1_diff = diff(err,b1);
err_b2_diff = diff(err,b2);

%% Training Process
iteration_num = 1000;
learning_rate = 0.01;

param_old = param_init;
param_new = zeros(length(param_old),1);

for k = 1:iteration_num
    % w1
    err_de = 0;
    for i = 1:length(x_train)
        err_de = err_de + subs(err_w1_diff,[x,y,w1,w2,w3,w4,w5,w6,b1,b2],[x_train(i),y_train(i),param_old(1),param_old(2),param_old(3),param_old(4),param_old(5),param_old(6),param_old(7),param_old(8)]);
    end
    err_de = double(err_de/length(x_train));
    param_new(1) = param_old(1) - learning_rate*err_de;
    
    % w2
    err_de = 0;
    for i = 1:length(x_train)
        err_de = err_de + subs(err_w2_diff,[x,y,w1,w2,w3,w4,w5,w6,b1,b2],[x_train(i),y_train(i),param_old(1),param_old(2),param_old(3),param_old(4),param_old(5),param_old(6),param_old(7),param_old(8)]);
    end
    err_de = double(err_de/length(x_train));
    param_new(2) = param_old(2) - learning_rate*err_de;
    
    % w3
    err_de = 0;
    for i = 1:length(x_train)
        err_de = err_de + subs(err_w3_diff,[x,y,w1,w2,w3,w4,w5,w6,b1,b2],[x_train(i),y_train(i),param_old(1),param_old(2),param_old(3),param_old(4),param_old(5),param_old(6),param_old(7),param_old(8)]);
    end
    err_de = double(err_de/length(x_train));
    param_new(3) = param_old(3) - learning_rate*err_de;

    % w4
    err_de = 0;
    for i = 1:length(x_train)
        err_de = err_de + subs(err_w4_diff,[x,y,w1,w2,w3,w4,w5,w6,b1,b2],[x_train(i),y_train(i),param_old(1),param_old(2),param_old(3),param_old(4),param_old(5),param_old(6),param_old(7),param_old(8)]);
    end
    err_de = double(err_de/length(x_train));
    param_new(4) = param_old(4) - learning_rate*err_de;

    % w5
    err_de = 0;
    for i = 1:length(x_train)
        err_de = err_de + subs(err_w5_diff,[x,y,w1,w2,w3,w4,w5,w6,b1,b2],[x_train(i),y_train(i),param_old(1),param_old(2),param_old(3),param_old(4),param_old(5),param_old(6),param_old(7),param_old(8)]);
    end
    err_de = double(err_de/length(x_train));
    param_new(5) = param_old(5) - learning_rate*err_de;

    % w5
    err_de = 0;
    for i = 1:length(x_train)
        err_de = err_de + subs(err_w6_diff,[x,y,w1,w2,w3,w4,w5,w6,b1,b2],[x_train(i),y_train(i),param_old(1),param_old(2),param_old(3),param_old(4),param_old(5),param_old(6),param_old(7),param_old(8)]);
    end
    err_de = double(err_de/length(x_train));
    param_new(6) = param_old(6) - learning_rate*err_de;

    % b1
    err_de = 0;
    for i = 1:length(x_train)
        err_de = err_de + subs(err_b1_diff,[x,y,w1,w2,w3,w4,w5,w6,b1,b2],[x_train(i),y_train(i),param_old(1),param_old(2),param_old(3),param_old(4),param_old(5),param_old(6),param_old(7),param_old(8)]);
    end
    err_de = double(err_de/length(x_train));
    param_new(7) = param_old(7) - learning_rate*err_de;

    % b2
    err_de = 0;
    for i = 1:length(x_train)
        err_de = err_de + subs(err_b2_diff,[x,y,w1,w2,w3,w4,w5,w6,b1,b2],[x_train(i),y_train(i),param_old(1),param_old(2),param_old(3),param_old(4),param_old(5),param_old(6),param_old(7),param_old(8)]);
    end
    err_de = double(err_de/length(x_train));
    param_new(8) = param_old(8) - learning_rate*err_de;

    param_old = param_new;
end

%% Test Model(s)
y_new = zeros(length(y_data),1);
for i = 1:length(x_data)
    y_new(i) = subs(out,[x,w1,w2,w3,w4,w5,w6,b1,b2],[x_data(i),param_old(1),param_old(2),param_old(3),param_old(4),param_old(5),param_old(6),param_old(7),param_old(8)]);
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