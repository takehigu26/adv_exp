import numpy as np
from sklearn.metrics import accuracy_score

def my_accuracy_score(y_true, y_pred):
    y_pred_modified = []
    for y in y_pred:
        y_pred_modified.append(np.argmax(y))
    y_pred_modified = np.array(y_pred_modified, 'int64')
    return accuracy_score(y_true, y_pred_modified)
