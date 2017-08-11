function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels (K).
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% =========================================================================

%%% MAP FROM INPPUT LAYER TO HIDDEN LAYER
% Add bias parameters into the 1st columns of eacho node in the input layer
a_1 = [ones(m, 1), X];
% Define the input operand to the second layer (hidden layer)
z_2 = a_1 * Theta1';
% Determine the weights via logistic regression
a_2 = sigmoid(z_2);


%%% MAP FROM HIDDEN LAYER TO OUTPUT LAYER
% Add bias parameters into the 1st columns of each node in the hidden layer
a_2 = [ones(m, 1), a_2];
% Define the input operand to the third layer (output layer) 
z_3 = a_2 * Theta2';
% Determine the output of the second layer [i.e. h_theta(x)]
a_3 = sigmoid(z_3);


% Determine the most likely category match for each example
[~, p] = max(a_3, [], 2);

% =========================================================================


end
