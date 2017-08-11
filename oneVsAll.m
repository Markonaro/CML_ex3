function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1); % number of training examples; # rows in X
n = size(X, 2); % number of pixels; # columns in X

% Rename num_labels to make code more concise & consistent w/ lectures
K = num_labels;

% You need to return the following variables correctly 
all_theta = zeros(K, n + 1); % # of different classifications (0-9) by
                             % # of pixels in each training example (20x20)

% Add ones to the X data matrix; i.e. add bias parameters
X = [ones(m, 1), X];

% =========================================================================

% Set Initial theta
initial_theta = zeros(n + 1, 1);

% Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 50);

for c = 1:K
        [theta] = ...
        fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
                initial_theta, options);
        % Set each row in the all_theta matrix equal to a column vector
        % of theta weights for that given example. y == c determines how
        % many times an entry y is equal to the current classification c. 
        all_theta(c, :) = theta';
end

% =========================================================================


end
