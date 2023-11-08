# author: Alexander Apers

import numpy as np
from typing import List
from tree_node import Tree, Node
from collections import deque
import fast_split

# Flag for using numba compilation to speed up the process of finding the optimal split.
USE_NUMBA_COMPILATION = True
np.random.seed(0)

def tree_grow(x: np.ndarray, y: np.ndarray, nmin: int, minleaf: int, nfeat: int) -> Tree:
    """
    Grow a single binary decision tree for numerical features.

    Name: tree_grow

    Args:
        x (np.ndarray): The training data matrix used to grow the tree on. Each row is datapoint.
        y (np.ndarray): The data labels for the training data.
        nmin (int): The minimum number of datapoints that is allowed for a node to still consider splitting it.
        minleaf (int): The minimum number datapoints for a valid leaf node.
        nfeat (int): The number of features that is considered for choosing the optimal split

    Returns:
        T (Tree): The binary decision tree grown on the training data (x, y).
    """


    assert 1 <= nfeat <= x.shape[1] 
    assert nmin >= 1
    assert minleaf >= 1

    root = Node(x, y)
    T = Tree(root)

    Q = deque()
    Q.appendleft(root)

    while len(Q) > 0:
        node = Q.pop()
        # only consider splitting a node if it has at least nmin samples and its impurity is larger than 0
        if node.impurity > 0 and len(node) >= nmin:
            features = select_random_features(x, nfeat)
            if USE_NUMBA_COMPILATION:
                best_split = fast_split.get_best_split_fast_numba(node.observations, node.labels, features, minleaf)
            else:
                best_split = node.get_best_split(features, minleaf)
            if best_split != None:
                left_node, right_node = T.apply_split(node, best_split)
                Q.appendleft(left_node)
                Q.appendleft(right_node)

    return T


def tree_pred(x: np.ndarray, tr: Tree) -> np.ndarray: 
    """
    Use a single binary decision tree for prediction on (new) data.

    Name: tree_pred

    Args:
        x (np.ndarray): A data matrix with datapoints that will be used for prediction. Each row is datapoint.
        tr (Tree): A binary decision tree that has already been trained.

    Returns:
        y (np.ndarray): The class predictions for all datapoints in x of tree tr.
    """

    y = -np.ones(x.shape[0])
    for i in range(x.shape[0]):
        y[i] = tr.predict(x[i])

    return y


def tree_grow_b(x: np.ndarray, y: np.ndarray, nmin: int, minleaf: int, nfeat: int, m: int) -> List[Tree]:
    """
    Create a list of m binary decision tree for numerical features trained on bootstrapped data samples.

    Name: tree_grow_b

    Args:
        x (np.ndarray): The training data matrix used to grow the trees on. Each row is datapoint.
        y (np.ndarray): The data labels for the training data.
        nmin (int): The minimum number of datapoints that is allowed for a node to still consider splitting it.
        minleaf (int): The minimum number datapoints for a valid leaf node.
        nfeat (int): The number of features that is considered for choosing the optimal split
        m (int): The number of trees that are grown on bootstrapped data samples.

    Returns:
        trees (List[Tree]): The list of m binary decision trees grown on the training data (x, y).
    """

    trees = []
    for _ in range(m):
        # draw bootstrapped datasamples of the same size as original dataset using replacement
        indices = np.random.choice(x.shape[0], size=x.shape[0], replace=True)
        T = tree_grow(x[indices], y[indices], nmin=nmin, minleaf=minleaf, nfeat=nfeat)
        trees.append(T)
    
    return trees


def tree_pred_b(x: np.ndarray, trs: List[Tree]) -> np.ndarray: 
    """
    Use a list of binary decision tree for prediction on (new) data using majority vote.

    Name: tree_pred_b

    Args:
        x (np.ndarray): A data matrix with datapoints that will be used for prediction. Each row is datapoint.
        trs (List[Tree]): A list of binary decision trees that have already been trained.

    Returns:
        y (np.ndarray): The majority class predictions for all datapoints in x of trees in trs.
    """

    y = np.zeros(x.shape[0])
    for tree in trs:
        y += tree_pred(x, tree)
    return np.where(y / len(trs) < .5, 0, 1)


def select_random_features(x: np.ndarray, nfeat: int) -> np.ndarray:
    """
    Returns the sorted indices of the features to be considered as a 1D array.

    Name: select_random_features

    Args:
        x (np.ndarray): A data matrix with datapoints that will be used for prediction. Each row is datapoint.
        nfeat (int): The number of features that is considered for choosing the optimal split

    Returns:
        selected_features (np.ndarray): The indices corresponding to the features that will be considered for a split.
    """

    return np.sort(np.random.choice(np.arange(x.shape[1]), size=nfeat, replace=False))
