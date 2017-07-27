function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
% =========================================================================
m = length(y); % number of training examples

% Defining the logit output per usual
ghX = sigmoid(X*theta);

% Defining the cost of log. reg. for all examples in training set.
J = (-1/m)*sum(y.*log(ghX)+(1-y).*log(1-ghX));

% Defining the instantaneous rate of error for the given parameters
grad = (1/m)*X'*(ghX-y);


% Rregularizing all weigths of the error function & updating overall cost.
J = J + (lambda/(2*m))*sum(theta(2:end).^2);

% Knowing we don't regularize the bias parameter for determining
% the gradient, we overwrite theta(1) = 0 and increment each
% parameter such that theta(i) = theta(i+1).
theta = [0; theta(2:end)];

% Regularizing the gradient of the error of each example 
% in the training set.
grad = grad + (lambda/m)*theta;

% =========================================================================

grad = grad(:);

end
