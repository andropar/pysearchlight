import numpy as np
import itertools
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


def fit_clf(data, labels, clf=SVC()):
    """Example searchlight function using a classifier.

    Parameters
    ----------
    data : np.ndarray
        Data from one searchlight sphere with shape ``(n_voxels, n_samples)``.
    labels : np.ndarray
        Array of labels for the samples.
    clf : sklearn.base.BaseEstimator, optional
        Classifier instance used for cross validation.

    Returns
    -------
    float
        Mean cross-validation accuracy of ``clf`` for the given data.
    """
    data = np.nan_to_num(data)

    cv_accuracies = cross_val_score(
        clf, data, labels, cv=5, n_jobs=1, scoring="accuracy"
    )

    return cv_accuracies.mean()
