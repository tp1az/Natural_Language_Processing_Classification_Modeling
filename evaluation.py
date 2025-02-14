"""Evaluation Metrics

Author: Kristina Striegnitz and Anthony Piacentini

I affirm that I have carried out my academic endeavors with full
academic honesty. Anthony Piacentini

Complete this file for part 1 of the project.
"""

def get_accuracy(y_pred, y_true):
    """Calculate the accuracy of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    correct_predicitons = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            correct_predicitons += 1
    accuracy = correct_predicitons / len(y_pred)
    return accuracy

def get_precision(y_pred, y_true, label=1):
    """Calculate the precision of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    label: label for which we are calculating precision
    """
    true_positive = 0
    false_positive = 0
    
    for i in range(len(y_pred)):
        if y_pred[i] == label:
            if y_true[i] == label:
                true_positive += 1
            else:
                false_positive += 1
    
    if true_positive + false_positive == 0:
        return 0  
    
    precision = true_positive / (true_positive + false_positive)
    return precision


def get_recall(y_pred, y_true, label=1):
    """Calculate the recall of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    label: label for which we are calculating precision
    """
    true_positive = 0
    false_negative = 0
        
    for i in range(len(y_true)):
        if y_true[i] == label:
            if y_pred[i] == label:
                true_positive += 1
            else:
                false_negative += 1

    if true_positive + false_negative == 0:
        return 0  # Avoid division by zero
    
    recall = true_positive / (true_positive + false_negative)
    return recall


def get_fscore(y_pred, y_true, label=1):
    """Calculate the f-score of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    label: label for which we are calculating precision
    """
    precision = get_precision(y_pred, y_true, label)
    recall = get_recall(y_pred, y_true, label)
    if precision + recall == 0:
        return 0  
        
    fscore = 2 * (precision * recall) / (precision + recall)
    return fscore



def evaluate(y_pred, y_true, label=1):
    """Calculate precision, recall, and f-score of the predicted labels
    and print out the results.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    print("Accuracy: {:.0f}%".format(get_accuracy(y_pred, y_true) * 100))
    print("Precision: {:.0f}%".format(get_precision(y_pred, y_true, label) * 100))
    print("Recall: {:.0f}%".format(get_recall(y_pred, y_true, label) * 100))
    print("F-score: {:.0f}%".format(get_fscore(y_pred, y_true, label) * 100))


if __name__ == "__main__":
    # Test the evaluation functions
    y_true = [1,1,1,1,1,0,0,0,0,0]
    y_pred = [1,1,0,0,1,0,0,1,0,0]
    evaluate(y_pred, y_true, 1)