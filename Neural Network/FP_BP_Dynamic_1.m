%{
    * Creator:          Chuong Nguyen
    * Email:            chuong19111996@gmail.com    
                        nguyenhoangkhanhchuong1996@gmail.com
    * File:             
    * Description:      
%}

%% Import Data & Separate Dataset
% Read data
data = readmatrix('hoorn_0dot1rad_50s_0dot0sample.xlsx');
start_i = 1;
end_i = 1700;
time = data(start_i:end_i,1);
delta = data(start_i:end_i,21);             % input #1 of NN
r = data(start_i:end_i,15);                 % output of NN
r_back1 = [r(1);r(1:(length(r)-1))];        % input #2 of NN
r_back2 = [r(1);r(1);r(1:(length(r)-2))];   % input #3 of NN

% plot again to check
figure(1)
plot(time, r);

% plot r again to check 
% plot_lim = 5;
% figure(1)
% hold on
% plot(time(1:plot_lim), r(1:plot_lim));
% plot(time(1:plot_lim), r_back1(1:plot_lim));
% plot(time(1:plot_lim), r_back2(1:plot_lim));
% legend('r','r_back1','r_back2');
hold off
% delta_cmd = data(:,24);

%% Parameters
% Input to hidden layer

% Hidden layer to output layer

param_init = zeros(11,1);

%% Network Structure

% Define symbolic variables
syms y out b_in b_out       % feature input, measured output, output, biases
x = sym('x',[1 2]);         % in this particular example, we have 4 inputs

w_in = sym('w_in',[1 6]);     % symbolic weights
                            % 3 neuron in hidden layer, 2 inputs, 1 output
                            % each neuron has 2 weights from each input
                            % so 3x2 = 6 is number of weights from input
                            % to hidden layer.
          
w_out = sym('w_out',[1 3]); % only 1 output, so 3x1 = 3 is number of
                            % weights from hidden layer to output
                            
zh = sym('zh',[1 3]);       % symbolic input func to hidden layer
h = sym('h',[1 3]);         % symbolic activation func (input to hidden layer)

% Input to hidden layer
index = 1;                  % index for zh() and h()
for i = 1:length(x):length(w_in)    % scan whole w_in with the jump of input size
    % 1 to 2: neuron #1 in hidden layer
    % 3 to 4: neuron #2 in hidden layer
    % 5 to 6: neuron #3 in hidden layer
    zh(index) = b_in;   % initialize zh(index) as bias first
    for j = 1:1:length(x)
        zh(index) = zh(index) + w_in(i+(j-1))*x(j); % then add all the term
    end
    %  Activation func at hidden layer
    h(index) = sigmoid_act_func(zh(index));
    index = index + 1; 
end

% Hidden layer to output layer
z_out = b_out; % initialize z_out as bias first
for i = 1:length(w_out)
    z_out = z_out + w_out(i)*h(i);
end

% Activation func at output layer
out = z_out*1.0;      % Use linear activation.

% Cost function
err = (1/2)*(y-out)^2;

%  Partial Derivative of Cost function w.r.t. w_in
err_diff_w_in = sym('err_diff_w_in', [1 length(w_in)]);
for i = 1:length(w_in)
    err_diff_w_in(i) = diff(err,w_in(i));
end

%  Partial Derivative of Cost function w.r.t. w_out
err_diff_w_out = sym('err_diff_w_out', [1 length(w_out)]);
for i = 1:length(w_out)
    err_diff_w_out(i) = diff(err,w_out(i));
end

%  Partial Derivative of Cost function w.r.t. biases
err_diff_b_in = diff(err,b_in);
err_diff_b_out = diff(err,b_out);

%% Training Process
iteration_num = 10;
learning_rate = 0.02;

param_old = param_init;
% We want param_new to have length sum of weights & biases, so just use a
% variable that have that, which is param_old in this case
param_new = zeros(length(param_old),1); 

for k = 1:iteration_num
    % Separate param_old into weights & biases to substitute to symbolic
    % function. 
    w_in_param = param_old(1:6); w_in_param = w_in_param';
    w_out_param = param_old(7:9); w_out_param = w_out_param';
    b_in_param = param_old(10); b_in_param = b_in_param';
    b_out_param = param_old(11); b_out_param = b_out_param';
    
    % for all w_in weights
    for i = 1:length(w_in_param)
        err_de = 0;
        for j = 1:length(time)
            x_in = [delta(j) r_back1(j)];
            y_in = r(j);
            err_de = err_de + subs(err_diff_w_in(i),[x,y,w_in,w_out,b_in,b_out],[x_in,y_in,w_in_param,w_out_param,b_in_param,b_out_param]);
        end
        err_de = double(err_de)/j;
        param_new(i) = param_old(i) - learning_rate*err_de;
    end
    
    % for all w_out weights
    for i = 1:length(w_out_param)
        err_de = 0;
        for j = 1:length(time)
            x_in = [delta(j) r_back1(j)];
            y_in = r(j);
            err_de = err_de + subs(err_diff_w_out(i),[x,y,w_in,w_out,b_in,b_out],[x_in,y_in,w_in_param,w_out_param,b_in_param,b_out_param]);
        end
        err_de = double(err_de)/j;
        param_new(i + 6) = param_old(i + 6) - learning_rate*err_de;
    end
    
    % for b_in
    err_de = 0;
    for j = 1:length(time)
        x_in = [delta(j) r_back1(j)];
        y_in = r(j);
        err_de = err_de + subs(err_diff_b_in,[x,y,w_in,w_out,b_in,b_out],[x_in,y_in,w_in_param,w_out_param,b_in_param,b_out_param]);
    end
    err_de = double(err_de)/j;
    param_new(10) = param_old(10) - learning_rate*err_de;
    
    % for b_out
    err_de = 0;
    for j = 1:length(time)
        x_in = [delta(j) r_back1(j)];
        y_in = r(j);
        err_de = err_de + subs(err_diff_b_out,[x,y,w_in,w_out,b_in,b_out],[x_in,y_in,w_in_param,w_out_param,b_in_param,b_out_param]);
    end
    err_de = double(err_de)/j;
    param_new(11) = param_old(11) - learning_rate*err_de;
    
    % update parameters
    param_old = param_new;
end

%% Test Model(s)
y_new = zeros(length(r),1);
w_in_param = param_old(1:6); w_in_param = w_in_param';
w_out_param = param_old(7:9); w_out_param = w_out_param';
b_in_param = param_old(10); b_in_param = b_in_param';
b_out_param = param_old(11); b_out_param = b_out_param';

for i = 1:length(time)
    x_in = [delta(j) r_back1(j)];
    y_new(i) = subs(out,[x,w_in,w_out,b_in,b_out],[x_in,w_in_param,w_out_param,b_in_param,b_out_param]);
end
figure
hold on
plot(x_data,y_data)
plot(x_data,y_new)
hold off




