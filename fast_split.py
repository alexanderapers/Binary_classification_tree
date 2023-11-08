import numpy as np
from typing import Tuple
from numba import njit

@njit
def get_best_split_fast_numba(x: np.ndarray, y: np.ndarray, feature_indices: np.ndarray, minleaf: int) -> Tuple[int, float]:
    """
    An algorithm for calculating the optimal split given a set of features and minleaf constraints sped up using numba compilation.

    Name: get_best_split_fast_numba

    Args:
        x (np.ndarray): Observations that will have to be split. Each row is datapoint.
        y (np.ndarray): The corresponding data labels.
        feature_indices (np.ndarray): The feature indices of features that should be considered for the split.
        minleaf (int): The minimum number datapoints for a valid leaf node.

    Returns:
        best_split (Tuple[int, float]): The optimal split that adheres to the constraints represented as a tuple of
            feature index and split value.
    """
    
    # use a numba compatible way of sorting the observations along all features
    x_numba = np.empty_like(x, dtype=np.int64)
    for j in range(x_numba.shape[1]):
        x_numba[:, j] = np.argsort(x[:, j])
    
    sorted_indices = x_numba
    n = x.shape[0]

    best_impurity = np.inf
    best_split = None

    # loop through every feature that needs to be considered 
    for feature_index in feature_indices:
        # keep track of the number of datapoints and number of datapoints with label 1 in left and right subtrees
        # start with all datapoints in the right subtree
        n_left, n_1_left, n_right, n_1_right = 0, 0, n, np.sum(y)

        # one by one move the sorted datapoints to the left subtree 
        for i, data_index in enumerate(sorted_indices[:, feature_index]):
            # update the trackers of the number of datapoints and datapoints with label 1
            n_left += 1
            n_right -= 1
            if y[data_index] == 1:
                n_1_left += 1
                n_1_right -= 1

            # check if the current split being considered passes the minleaf constraint 
            if n_left < minleaf or n_right < minleaf:
                continue
            
            # calculate the current feature value and the next feature value in the data
            x_value = x[data_index, feature_index]
            next_data_index = sorted_indices[i+1, feature_index]
            x_next_value = x[next_data_index, feature_index]

            # only proceed with the split if the current and next feature values are distinct.
            # it's not possible to split between these points.
            if x_value == x_next_value:
                continue

            # calculate the split value and the corresponding weighted impurity of the split
            split_value = (x_value + x_next_value) / 2
            impurity = n_1_left * (1 - n_1_left / n_left) + n_1_right * (1 - n_1_right / n_right)

            # update the current best split if we find a lower impurity than the current best
            if impurity >= best_impurity:
                continue

            best_impurity = impurity
            best_split = (feature_index, split_value)

    return best_split


