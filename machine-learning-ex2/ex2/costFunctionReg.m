function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);

term1 =  -y' * log(h);
term2 = (1 - y)' * log(1 - h);
cost = (term1 - term2) / m;
withoutTheta1 = theta;
withoutTheta1(1, 1) = 0;
regularizationTerm = (lambda / (2 * m)) * (withoutTheta1' * withoutTheta1);
J = cost + regularizationTerm;

regularizationVector = (lambda / m) * theta;

% Don't regularize term theta(1)
regularizationVector(1, 1) = 0;

grad = (X' * ((h - y)) / m) + regularizationVector;

% =============================================================

end
