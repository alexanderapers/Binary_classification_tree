import numpy as np
from typing import List, Tuple, Dict

class Node:
    """
    Represents the nodes in a decision tree and the corresponding training data points.

    Name: Node

    Args:
        x (np.ndarray): Observations that arrive in this node in decision tree. Each row is datapoint.
        y (np.ndarray): The corresponding data labels for x.

    Members:
    self.observations (np.ndarray): Observations that arrive in this node in decision tree. Each row is datapoint.
    self.labels (np.ndarray): The corresponding data labels for x.
    self.parent (Node): The parent node of the node in the decision tree.
    self.children (Tuple[Node, Node]): None if the node is a leaf node, or the children nodes of the node as a tuple otherwise.
    self.split (Tuple[int, float]): None if the node is a leaf node, or the feature index and split value as a tuple otherwise.
    """

    observations: np.ndarray
    labels: np.ndarray
    parent: 'Node'
    children: Tuple['Node', 'Node']
    split: Tuple[int, float]

    def __init__(self, x, y):
        self.observations = x
        self.labels = y
        self.parent = None
        self.children = None
        self.split = None
        

    def __len__(self):
        return self.observations.shape[0]
    

    def __str__(self):
        # string representation of the split in the node using feature index and split value
        if self.split != None:
            return str(self.split[0]) + ": " + str(self.split[1])
        else:
            return "leaf"


    def split_string_representation(self, feature_names: Dict[int, str]) -> str:
        """
        Given a mapping from feature indices to names creates a string representation of the split in the node.

        Name: Node.split_string_representation

        Args:
            feature_names (Dict[int, str]): A mapping from feature indices to feature names as strings.

        Returns:
            split_representation (str): The string representation of the split in the node.
        """

        return f"|{str(feature_names[self.split[0]])} < {str(self.split[1])}|"
    
    
    def class_labels_string_representation(self) -> str:
        """
        Creates a string representation of the distribution of class labels in the node.

        Name: Node.class_labels_string_representation

        Returns:
            split_representation (str): The string representation of the class distribution in the node.
        """
          
        return f"|#0:{self.zeros},#1:{self.ones}|"


    @property
    def majority_class(self):
        # Returns the majority class of observations in the node.
        return np.sum(self.labels) > (len(self.labels) / 2)


    @property
    def impurity(self):
        # Returns the impurity (calculated with gini) of observations in the node.
        return self.gini_index()
    
    
    @property
    def ones(self):
        # Returns the number of datapoints with 1 as label.
        return np.sum(self.labels)


    @property
    def zeros(self):
        # Returns the number of datapoints with 0 as label.
        return len(self) - self.ones
    

    def gini_index(self):
        """
        Calculates the gini index of observations in the node.

        Name: Node.gini_index

        Returns:
            gini (float): The gini index of observations in the node.
        """
        mean = np.mean(self.labels)
        return mean * (1 - mean)    


    def get_best_split(self, feature_indices: np.ndarray, minleaf: int) -> Tuple[int, float]:
        """
        An algorithm for calculating the optimal split given a set of features and minleaf constraints.

        Name: Node.get_best_split

        Args:
            feature_indices (np.ndarray): The feature indices of features that should be considered for the split.
            minleaf (int): The minimum number datapoints for a valid leaf node.

        Returns:
            best_split (Tuple[int, float]): The optimal split that adheres to the constraints represented as a tuple of
                feature index and split value
        """
          
        x = self.observations
        y = self.labels
        # determine the indices that would sort the datapoints on every feature
        sorted_indices = np.argsort(x, axis=0)
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
    

    def predict(self, x: np.ndarray) -> bool:
        """
        Recursively predict the class label for the single datapoint that comes across the node.

        Name: Node.predict

        Args:
            x (np.ndarray): A single datapoints for which a class label has to be predicted.

        Returns:
            class_label (bool): The class label that is predicted by the tree.
        """
          
        # if the node is a leaf then use the majority class as a prediction
        if self.children == None:
            return self.majority_class
        
        # otherwise recurse into the left or right subtree based on the split of the node
        elif x[self.split[0]] < self.split[1]:
            return self.children[0].predict(x)
        else:
            return self.children[1].predict(x)


    def get_best_split_slow(self, feature_indices: np.ndarray, minleaf: int) -> Tuple[int, float]:
        """
        An slow and old implementation for calculating the optimal split given a set of features and minleaf constraints.
        It's not used in the code anymore.

        Name: Node.get_best_split_slow

        Args:
            feature_indices (np.ndarray): The feature indices of features that should be considered for the split.
            minleaf (int): The minimum number datapoints for a valid leaf node.

        Returns:
            best_split (Tuple[int, float]): The optimal split that adheres to the constraints represented as a tuple of
                feature index and split value
        """
         
        best_impurity_reduction = float("-inf")
        best_split = None

        # sort every column independently on feature value 
        sorted_observations = np.sort(self.observations, axis=0)
        # create all the possible split values for every feature
        split_values = (sorted_observations[:-1, :] + sorted_observations[1:, :]) / 2
        
        # only consider features that have been randomly selected
        for feature_index in feature_indices:
            # get unique split values of feature
            splits = np.unique(split_values[:, feature_index])
            # determine which of these splits is allowed
            splits = self.splits_allowed(feature_index, splits, minleaf)

            # determine the impurity reduction of all
            for split in splits:
                mask = self.observations[:, feature_index] < split
                sum_left = np.sum(mask)

                # it's slow because it's explicitly splitting the data for every single split value and 
                # recalculating the impurity from scratch
                ml = np.mean(self.labels[mask])
                mr = np.mean(self.labels[~mask])

                impurity_left = ml * (1 - ml)
                impurity_right = mr * (1 - mr)
                impurity_reduction = self.impurity - (sum_left * impurity_left + (len(self)-sum_left)  * impurity_right) / len(self)

                if impurity_reduction < best_impurity_reduction:
                    continue

                best_impurity_reduction = impurity_reduction
                best_split = (feature_index, split)        

        return best_split
    


class Tree:
    """
    Represents the structure of the decision tree itself.

    Name: Tree

    Args:
        root (Node): The root node that starts the decision tree.

    Members:
    self.root (Node): The root node that starts the decision tree.
    """

    root: Node

    def __init__(self, root: Node):
        self.root = root


    def __str__(self):
        # create a string representation of the tree by traversing the nodes of the tree in preorder order
        result = self.dfs_preorder([], self.root)
        return " ".join(result).rstrip(" ^")
    

    def dfs_preorder(self, result: List[str], node: Node) -> List[str]:
        """
        Performs a dfs in preorder to accumulate the string representation of the tree.

        Name: Tree.dfs_preorder

        Args:
            result (List[str]): The result string accumulated up to this point.
            node (Node): The node that we're currently at in the traversal.

        Returns:
            result (List[str]): The result string accumulated up to this point.
        """

        # append the string representation of the current node
        result.append("|" + str(node) + "|")
        # and continue the traversal if the node has children
        if node.children != None:
            self.dfs_preorder(result, node.children[0])
            self.dfs_preorder(result, node.children[1])
            # append ^ to signify going up in the tree when a node doesn't have any more children
            result.append("^")
        else:
            result.append("^")   
        return result 

    
    def apply_split(self, parent: Node, split: Tuple[int, float]) -> Tuple[Node, Node]:
        """
        Once an optimal split given the constraints has been found it is applied to the tree.

        Name: Tree.apply_split

        Args:
            parent (Node): The parent node that is about to receive children.
            split (Tuple[int, float]): The optimal split given constraints that has been determined for the node.

        Returns:
            (left, right) (Tuple[Node, Node]): The left and right child nodes of parent that have just been generated.
        """

        parent.split = (split[0], split[1])
        # determine which observations go in the left and right subtree respectively
        mask = parent.observations[:, split[0]] < split[1]
        # and create the left and right child nodes and update the pointers
        left = Node(parent.observations[mask], parent.labels[mask])
        right = Node(parent.observations[~mask], parent.labels[~mask])
        parent.children = (left, right)
        left.parent = right.parent = parent

        return left, right


    def predict(self, x: np.ndarray) -> bool:
        """
        Recursively predict the class label for the single datapoint.

        Name: Tree.predict

        Args:
            x (np.ndarray): A single datapoints for which a class label has to be predicted.

        Returns:
            class_label (bool): The class label that is predicted by the tree.
        """
          
        return self.root.predict(x)
    

    def pruned_subtree_string(self, feature_names: Dict[int, str]) -> str:
        """
        Creates a string representation of the first three splits (if they exist) of a tree given a mapping from 
            feature indices to feature names.

        Name: Tree.pruned_subtree_string

        Args:
            feature_names (Dict[int, str]): A mapping from feature indices to feature names as strings.

        Returns:
            result (str): String repreentation of the first three splits in the tree with feature names.
        """
          
        result = []
        distributions = []
        a = self.root
        result.append(a.split_string_representation(feature_names)) 
        distributions.append(a.class_labels_string_representation())
        if self.root.children != None:
            b = self.root.children[0]
            result.append(b.split_string_representation(feature_names)) 
            distributions.append(b.class_labels_string_representation())
            if b.children != None:
                c, d = b.children
                result.append("|leaf| ^ |leaf| ^ ^")
                distributions.append(c.class_labels_string_representation())
                distributions.append(d.class_labels_string_representation())
            e = self.root.children[1]
            result.append(e.split_string_representation(feature_names))
            distributions.append(e.class_labels_string_representation())
            if e.children != None:
                f, g = e.children
                result.append("|leaf| ^ |leaf|")
                distributions.append(f.class_labels_string_representation())
                distributions.append(g.class_labels_string_representation())

        return " ".join(result) + "\n" + " ".join(distributions)

    
