import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    accuracy = 1 - np.sum(np.abs(np.logical_xor(prediction, ground_truth))) / len(prediction) if len(prediction) != 0 else 1
    
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for i in range(len(prediction)):
        if prediction[i] and ground_truth[i]:
            true_positive += 1
        elif not prediction[i] and not ground_truth[i]:
            true_negative += 1
        elif prediction[i] and not ground_truth[i]:
            false_positive += 1
        else:
            false_negative += 1
            
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    trues = []
    for i in range(len(prediction)):
        trues.append(prediction[i] == ground_truth[i])
    return np.sum(trues) / len(prediction) if len(prediction) != 0 else 0
