import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.layer_1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu_1 = ReLULayer()
        self.layer_2 = FullyConnectedLayer(hidden_layer_size, n_output)
        

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        for param in self.params().values():
            param.grad = np.zeros_like(param.value)
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        pred = self.layer_1.forward(X)
        pred = self.relu_1.forward(pred)
        pred = self.layer_2.forward(pred)
        
        loss, grad = softmax_with_cross_entropy(pred, y)
        
        grad = self.layer_2.backward(grad)
        grad = self.relu_1.backward(grad)
        grad = self.layer_1.backward(grad)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for param in self.params().values():
            reg_loss, reg_grad = l2_regularization(param.value, self.reg)
            param.grad += reg_grad
            loss += reg_loss

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = self.layer_1.forward(X)
        pred = self.relu_1.forward(pred)
        pred = self.layer_2.forward(pred)
        return np.argmax(pred, axis=1)

    def params(self):
        result = {}

        result["W1"] = self.layer_1.params()["W"]
        result["B1"] = self.layer_1.params()["B"]
        result["W2"] = self.layer_2.params()["W"]
        result["B2"] = self.layer_2.params()["B"]

        return result
