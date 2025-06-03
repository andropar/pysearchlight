import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from typing import Union, List
from numpy.typing import NDArray


def fit_clf(
    data: NDArray[np.floating],
    labels: Union[NDArray, List[NDArray]],
    clf: SVC = SVC(),
) -> float:
    """
    Returns classification accuracy for a single voxel (cross-validated). Labels can either be
    a numpy array or a list of numpy arrays (in which case the classification accuracies for every
    label array in the list is returned).
    """
    data = np.nan_to_num(data)

    cv_accuracies = cross_val_score(
        clf, data, labels, cv=5, n_jobs=1, scoring="accuracy"
    )

    return cv_accuracies.mean()
