import numpy as np
from sklearn.metrics import accuracy_score
# y_true = array([0, 1, 1, ..., 0]) y_pred = array([[0.46, 0.54], [0.92, 0.08], ...])
def my_accuracy_score(y_true, y_pred):
    # number of patterns for label
    n_patterns = len(y_pred[0])
    y_pred_modified = []
    for y in y_pred:
        y_pred_modified.append(np.argmax(y))
    y_pred_modified = np.array(y_pred_modified, 'int64')
    return accuracy_score(y_true, y_pred_modified)
