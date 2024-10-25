# import numpy as np
# from collections import Counter

# class Node:
#     def __init__(self, feature=None, threshold=None, left=None, right=None, information_gain=None, value=None):
#         self.feature = feature
#         self.threshold = threshold
#         self.left = left
#         self.right = right
#         self.information_gain = information_gain
#         self.value = value

#     def print_attributes(self):
#         print("Feature of the split: " + str(self.feature))
#         print("Threshold of the split: " + str(self.threshold))
#         print("Left Child: " + str(self.left))
#         print("Right Child: " + str(self.right))
#         print("Information Gain: " + str(self.information_gain))
#         print("Value of the node: " + str(self.value))

#     def is_leaf_node(self):
#         return self.value is not None


# class DecisionTree:
#     def __init__(self, num_features, min_sample_split=2, max_depth=100):
#         self.min_sample_split = min_sample_split
#         self.max_depth = max_depth
#         self.num_features = num_features
#         self.root = None
        
#     def fit(self, X, y):
#         self.root = self._grow_tree(X, y)

#     def _grow_tree(self, X, y, depth=0):
#         num_samples, num_feats = X.shape
#         num_labels = len(np.unique(y))

#         if (depth > self.max_depth or num_labels == 1 or num_samples < self.min_sample_split):
#             leaf_value = self._most_common_label(y)
#             return Node(value=leaf_value)

#         feat_index = np.random.choice(num_feats, self.num_features, replace=False)
#         best_feature, best_threshold = self._best_split(X, y, feat_index)

#         # Handle case where no valid split was found
#         if best_feature is None:
#             leaf_value = self._most_common_label(y)
#             return Node(value=leaf_value)

#         left_idx, right_idx = self._split(X[:, best_feature], best_threshold)

#         # Check if left or right split is empty
#         if len(left_idx) == 0 or len(right_idx) == 0:
#             leaf_value = self._most_common_label(y)
#             return Node(value=leaf_value)

#         left = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
#         right = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)

#         return Node(best_feature, best_threshold, left, right)

#     def _best_split(self, X, y, feat_index):
#         best_gain = -1
#         split_index, split_threshold = None, None

#         for feat_idx in feat_index:
#             X_column = X[:, feat_idx]
#             thresholds = np.unique(X_column)

#             for thresh in thresholds:
#                 gain = self._information_gain(y, X_column, thresh)

#                 if gain > best_gain:
#                     best_gain = gain
#                     split_index = feat_idx
#                     split_threshold = thresh

#         # Return None if no valid split was found
#         if best_gain == -1:
#             return None, None

#         return split_index, split_threshold
    
#     def _information_gain(self, y, X_column, threshold):
#         parent_entropy = self._entropy(y)
#         left_idx, right_idx = self._split(X_column, threshold)

#         # Avoid division by zero if either split is empty
#         if len(left_idx) == 0 or len(right_idx) == 0:
#             return 0

#         n = len(y)
#         n_left, n_right = len(left_idx), len(right_idx)
#         e_left, e_right = self._entropy(y[left_idx]), self._entropy(y[right_idx])

#         child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
#         information_gain = parent_entropy - child_entropy
#         return information_gain

#     def _split(self, X_column, split_threshold):
#         left_idxs = np.argwhere(X_column <= split_threshold).flatten()
#         right_idxs = np.argwhere(X_column > split_threshold).flatten()
#         return left_idxs, right_idxs
    
#     def _entropy(self, y):
#         hist = np.bincount(y)
#         ps = hist / len(y)
#         entropy = 0  # Initialize entropy
#         for p in ps:
#             if p > 0:
#                 entropy += p * np.log2(p)
#         return -entropy

#     def _most_common_label(self, y):
#         if len(y) == 0:  # Added check for empty array
#             return None  # Handle this case as you see fit
#         counter = Counter(y)
#         value = counter.most_common(1)[0][0]
#         return value
    
#     def predict(self, X):
#         return np.array([self._traverse_tree(x, self.root) for x in X])
    
#     def _traverse_tree(self, x, node):
#         if node.is_leaf_node():
#             return node.value
        
#         if x[node.feature] <= node.threshold:
#             return self._traverse_tree(x, node.left)
#         return self._traverse_tree(x, node.right)


import numpy as np
from collections import Counter
import pandas as pd

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, information_gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.information_gain = information_gain
        self.value = value
    
    def print_attributes(self):
        # This method prints the attributes of the given node
        print("Feature to be split:", self.feature)
        print("Threshold of split:", self.threshold)
        print("Left child:", self.left)
        print("Right child:", self.right)
        print("Information Gain:", self.information_gain)
        print("Value at current node:", self.value)


class DecisionTree:
    def __init__(self, n_features, feature_names, min_samples_split=50, max_depth=15):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.num_features = n_features  # Store number of features as integer
        self.feature_names = feature_names  # Store feature names

    def _entropy(self, y):
        count = np.bincount(np.array(y, dtype=np.int64))
        pb = count / len(y) 
        entropy = -np.sum([i * np.log2(i) for i in pb if i > 0])
        return entropy
        
    def _information_gain(self, parent_node, left_child_node, right_child_node):
        left_ratio = len(left_child_node) / len(parent_node)
        right_ratio = len(right_child_node) / len(parent_node)
        parent_entropy = self._entropy(parent_node)
        left_entropy = self._entropy(left_child_node)
        right_entropy = self._entropy(right_child_node)        
        information_gain = parent_entropy - ((left_entropy * left_ratio) + (right_entropy * right_ratio))
        return information_gain
    
    def _calculate_best_split(self, feature, label):
        best_split = {
            "feature_index": None,
            "threshold": None,
            "left": None,
            "right": None,
            "information_gain": -1
        }
        best_information_gain = -1
        (_, columns) = feature.shape

        for i in range(columns):
            x_current = feature[:, i]
            for threshold in np.unique(x_current):
                dataset = np.concatenate((feature, label.reshape(1, -1).T), axis=1)
                dataset_left = np.array([row for row in dataset if row[i] <= threshold])
                dataset_right = np.array([row for row in dataset if row[i] > threshold])

                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y = dataset[:, -1]
                    y_left = dataset_left[:, -1]
                    y_right = dataset_right[:, -1]
                    information_gain = self._information_gain(y, y_left, y_right)
                    if information_gain > best_information_gain:
                        best_split = {
                            "feature_index": i,
                            "threshold": threshold,
                            "left": dataset_left,
                            "right": dataset_right,
                            "information_gain": information_gain
                        }
                        best_information_gain = information_gain
        if best_split["feature_index"] is not None:
            print("Splitted Column:", self.feature_names[best_split['feature_index']])
        return best_split
    
    def _grow_tree(self, X, y, depth=0):
        num_rows, num_cols = X.shape
        print("-----------------------------")
        print(f"At Level {depth}:")
        print(f"Number of instances of X: {num_rows}")
        print(f"Number of columns to split in X: {num_cols}")
        print("------------------------------")
        
        condition1 = (num_rows >= self.min_samples_split)
        condition2 = (depth < self.max_depth)
        
        if condition1 and condition2:
            splitted_data = self._calculate_best_split(X, y)
            if splitted_data['information_gain'] > 0:
                new_depth = depth + 1
                print(f"Left Split to level: {new_depth}")
                X_left = splitted_data['left'][:, :-1]
                y_left = splitted_data['left'][:, -1]
                left_child = self._grow_tree(X_left, y_left, new_depth)
                
                print(f"Right Split to level: {new_depth}")
                X_right = splitted_data['right'][:, :-1]
                y_right = splitted_data['right'][:, -1]
                right_child = self._grow_tree(X_right, y_right, new_depth)
                
                return Node(
                    feature=splitted_data['feature_index'],
                    threshold=splitted_data['threshold'],
                    left=left_child,
                    right=right_child,
                    information_gain=splitted_data['information_gain']
                )
        
        return Node(value=Counter(y).most_common(1)[0][0])
              
    def fit(self, X, y):
        print("-----------------------------")
        print("Training Process Started.")
        self.root = self._grow_tree(X, y)

    def _predict(self, x, tree):
        if tree.value is not None:
            print(int(tree.value))
            return tree.value
        feature = x[tree.feature]
        
        if feature <= tree.threshold:
            print(f"Left Split: {self.feature_names[tree.feature]} <= {tree.threshold}")
            return self._predict(x=x, tree=tree.left)
        else:
            print(f"Right Split: {self.feature_names[tree.feature]} > {tree.threshold}")
            return self._predict(x=x, tree=tree.right)
     
    def predict(self, X, n_features):
        self.n_features = n_features
        return [self._predict(x, self.root) for x in X]

