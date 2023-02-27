%{
    * Creator:          Chuong Nguyen
    * Email:            chuong19111996@gmail.com    
                        nguyenhoangkhanhchuong1996@gmail.com
    * File:             Linear Regression from scratch
    * Description:      Linear Regression from scratch with simple model of
                        y = theta_0 + theta_1*x
                        where x has only one feature.
%}

%{
    * Key parts:
        (1) Import Data & Separate Dataset
        (2) Parameters
        (3) Define Cost function (used in Gradient Descent, not used directly)
        (4) Define Gradient Descent function
        (5) Train Data
        (6) Test Model(s)
%}
%% Clean
clc
clear

%% Import Data & Separate Dataset
% Read data
data = readmatrix('dataset1.xlsx');
x = data(:,1);
y = data(:,2);

% Separate data
ratio = 0.7;                            % Define separate ratio
datapt_num = length(data(:,1));         % Number of data points
trainpt_num = round(datapt_num*ratio);  % Rounded number of training points

% Training set
x_train = x(1:trainpt_num);
y_train = y(1:trainpt_num);
% figure
% plot(x_train,y_train);

% Validation set
x_valid = x(trainpt_num+1:end);
y_valid = y(trainpt_num+1:end);
% figure
% plot(x_valid,y_valid);

% Normalize Data

%% Parameters
% Linear model: y = theta_0 + theta_1*x
theta_0 = 2;                            % Define "randomly", in this example.
theta_1 = 3;                            % Define "randomly", in this example.
param_init = [theta_0,theta_1];         % Initial Parameters

%% Training Process
iteration_num = 1000;
learning_rate = 0.01;

param_old = param_init;
param_new = [0,0];
for k = 1:iteration_num
    param_new(1) = gradient_descent(param_old(1),learning_rate,x_train,y_train,param_old,0);
    param_new(2) = gradient_descent(param_old(2),learning_rate,x_train,y_train,param_old,1);
    param_old = param_new;
end

%% Test Model(s)
y_new = zeros(length(y),1);
for i = 1:length(x)
    y_new(i) = param_new(1) + param_new(2)*x(i);
end
figure
hold on
plot(x,y)
plot(x,y_new)
hold off

error = zeros(length(y),1);
for i = 1:length(x)
    error(i) = y(i) - y_new(i);
end
figure
plot(error)

%% Cost function

% Original cost function (not used)
% function J = cost_func(x,y,param)
%     % J(theta_0,theta_1) = (1/(2*m))*sum[(h(x(i))-y)^2]
%     sum = 0;
%     feature_size = length(x);
%     for i = 1:feature_size
%         h = param(1) + param(2)*x(i);
%         sum = sum + (h - y)^2;
%     end
%     J = sum/(2*feature_size);
% end

% Cost function with derivative, used in Gradient Descent
function J = cost_func_derivative(x,y,param,param_index)
    % J(theta_0,theta_1) = (1/(2*m))*sum[(h(x(i))-y)^2]
    J = 0;
    feature_size = length(x);
    syms theta_0 theta_1 x_syms y_syms
    h = theta_0 + theta_1*x_syms;
    J_part = (h - y_syms)^2;
    
    % Since J is derivative of a sum, we take derivate of each component of
    % the sum insead, which is J_part, then sum them all
    if param_index == 0
        J_part = diff(J_part,theta_0);
    else
        J_part = diff(J_part,theta_1);
    end
    
    % Sum for ALL data
    for i = 1:feature_size
        J_part = subs(J_part,[theta_0,theta_1,x_syms,y_syms],[param(1),param(2),x(i),y(i)]);
        J = J + J_part;
    end
    J = J/feature_size;
end

%% Gradient Descent function
function theta_j_new = gradient_descent(theta_j_old,rate,x,y,param,param_index)
    % theta_j_new = theta_j_old - rate*(dJ/dtheta_j)(theta)
    theta_j_new = theta_j_old - rate*cost_func_derivative(x,y,param,param_index);
end
