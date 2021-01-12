import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W

    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    predictions = predictions.copy()
    if predictions.ndim == 1:
        predictions -= np.max(predictions)
        exp_pred = np.exp(predictions)
        return exp_pred / np.sum(exp_pred)
    elif predictions.ndim == 2:
        predictions -= np.max(predictions, axis=1).reshape(-1, 1)
        exp_pred = np.exp(predictions)
        return exp_pred / np.sum(exp_pred, axis=1).reshape(-1, 1)
    else:
        raise Exception("Dims must be one or two")
        
        
def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    if probs.ndim == 1:
        return -np.log(probs[target_index])
    elif probs.ndim == 2:
        return - np.sum(np.log(probs[np.arange(probs.shape[0]), target_index]))
    else:
        raise Exception("Dims must be one or two")


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    target_resh = target_index.reshape(-1)
    probs = softmax(preds)
    loss = cross_entropy_loss(probs, target_resh)
    dprediction = None
    if probs.ndim == 1:
        probs[target_resh] -= 1
        dprediction = probs
    elif probs.ndim == 2:
        m = target_resh.shape[0]
        probs[range(m), target_resh] -= 1
        dprediction = probs
    else:
        raise Exception("Dims must be one or two")
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.mask = (X > 0)
        return X * self.mask

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = d_out * self.mask
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        return np.dot(self.X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute
        E = np.ones(shape=(1, self.X.shape[0]))
        self.B.grad = E.dot(d_out)
        self.W.grad = self.X.T.dot(d_out)
        # It should be pretty similar to linear classifier from
        # the previous assignment
        return d_out.dot(self.W.value.T)

    def params(self):
        return {'W': self.W, 'B': self.B}
