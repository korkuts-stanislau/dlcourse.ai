import numpy as np


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
    


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    target_resh = target_index.reshape(-1)
    probs = softmax(predictions)
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
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    loss, dpred = softmax_with_cross_entropy(predictions, target_index)
    dW = X.T.dot(dpred)
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            # TODO implement generating batches from indices
            loss = 0
            for batch_indices in batches_indices:
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                # Compute loss and gradients
                func_loss, func_grad = linear_softmax(X_batch, self.W, y_batch)
                reg_loss, reg_grad = l2_regularization(self.W, reg)
                batch_loss, batch_grad = func_loss + reg_loss, func_grad + reg_grad
                # Apply gradient to weights using learning rate
                loss += batch_loss
                self.W -= learning_rate * batch_grad
                # Don't forget to add both cross-entropy loss
                # and regularization!
            loss_history.append(loss)
            # end
            if (epoch + 1) % 10 == 0:
                print("Epoch %i, loss: %f" % (epoch + 1, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        pred = np.dot(X, self.W)
        y_pred = np.argmax(pred, axis=1).reshape(-1)
        return y_pred



                
                                                          

            

                
