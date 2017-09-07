    function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels0
 ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
y1=zeros(m,num_labels);
for i=1:m
    for j=1:num_labels
        if(j==y(i))
            y1(i,j)=1;
        end
    end
end
%adding the bias unit to the input
X = [ones(size(X, 1),1), X]; 
% computing the activation of the layer 1
Z1 = X * Theta1';
A1 = sigmoid(Z1);
% adding the bias unit to activation of layer 1
A1 = [ones(size(A1, 1),1), A1];
% computing the activation of the layer 2
Z2 = A1 * Theta2';
A2 = sigmoid(Z2);
% extracting the relevant columns from the respective Thetas
t1 = Theta1(:,2:size(Theta1,2));
t2 = Theta2(:,2:size(Theta2,2));
% computing the regularization term
reg = sum(sum((t1.^2)))+sum(sum((t2.^2)));
%computing the regularized function
J = (trace((- log(A2)' * y1  - log(1 - A2)' * (1 - y1)))/m) + (lambda/(2*m)) * (reg);
% computing error variable in layer 3
del3 = A2 - y1;
sigdel2 = [ones(1,m);(sigmoidGradient(Z1))'];
%computing error in second layer
del2 = (Theta2' * del3') .* sigdel2;
%Acculmulating the error
Del2=del3' * A1;
del2 = del2(2:end,:);
Del1=del2 * X;
%removing the bias unit interference from regularization
Theta1(:,1) = 0;
Theta2(:,1) = 0;
Theta1_grad = (1/m) * Del1 + lambda * Theta1/m; 
Theta2_grad = (1/m) * Del2 + lambda * Theta2/m;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
