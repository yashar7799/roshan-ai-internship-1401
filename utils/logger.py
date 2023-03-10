"""
Save Plots in experiment tracking (mlflow)
"""

import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from .mlflow_handler import MLFlowHandler


def get_logs(model, test_loader, unique_indexes, unique_labels, mlflow_handler: MLFlowHandler):

    """
    Parameters
    ----------
    model           A `Tensorflow.keras.Model` instance.
    test_loader     test DataGenerator
    mlflow_handler  a class, an interface to work easily with mlfow

    Returns
    -------
    Nothing         just save plots of predicted mask by model to visual intuition
    """

    print('Making  prediction of test data ...\n')

    y_pred, y_true = None, None
    for x, y in test_loader:
        predictions = model.predict(x)
        temp = np.argmax(predictions, axis=1)
        y = np.argmax(y, axis=1)
        if y_true is None:
            y_true = y
            y_pred = temp
        else:
            y_true = np.concatenate([y_true, y])
            y_pred = np.concatenate([y_pred, temp])

    correct_count = 0.0
    for i, y1 in enumerate(y_pred):
        if y1 == y_true[i]:
            correct_count = correct_count + 1
    mlflow_handler.add_report(f"Test Accuracy:  {correct_count / len(y_pred)}" + "\n", 'logs/test_accuracy.txt')
    print(f"Test Accuracy:  {correct_count / len(y_pred)}\n")
    # Metrics: Confusion Matrix
    con_mat = confusion_matrix(y_true, y_pred)
    print(con_mat)
    print('\n')
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index=[i for i in unique_labels], columns=[i for i in unique_labels])
    fig = plt.figure(figsize=(len(unique_labels), len(unique_labels)), tight_layout=True)
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()
    mlflow_handler.add_figure(fig, 'logs/confusion_matrix.png')
    report = classification_report(y_true, y_pred, labels=unique_indexes, target_names=unique_labels)
    mlflow_handler.add_report(report, 'logs/classification_report.txt')
    print(report)