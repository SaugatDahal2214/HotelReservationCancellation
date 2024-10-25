from decision_tree import DecisionTree
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_features, feature_names, n_trees=5, max_depth=15, min_samples_split=50):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.feature_names = feature_names  # Add feature names here
        self.trees = []
   
    def _sample(self, X, y):
        n_rows, n_cols = X.shape
        # sampling the dataset with replacements
        sample = np.random.choice(a = n_rows, size = n_rows, replace = True)
        samples_x = X[sample]
        samples_y = y[sample]
        return samples_x,samples_y
    
    
    def fit(self, X, y):
        i = 0
        if len(self.trees) > 0:
            self.trees = []
        tree_built = 0
        while tree_built < self.n_trees:
            print("-----------------------------")
            print("Iteration: {0}".format(i))
            # Pass both n_features and feature_names to each DecisionTree instance
            tree = DecisionTree(n_features=self.n_features, feature_names=self.feature_names, 
                                min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            sample_x, sample_y = self._sample(X, y)
            tree.fit(sample_x, sample_y)
            self.trees.append(tree)
            tree_built += 1
            i += 1
      
    
    def predict(self, X, n_features):
        self.n_features = n_features
        labels = []

    # Collect predictions from each tree in the forest
        for tree in self.trees:
            labels.append(tree.predict(X, n_features))
        
        # Transpose the list of labels and determine the majority vote for each instance
        labels = np.swapaxes(a=labels, axis1=0, axis2=1)
        predictions = []
        for preds in labels:
            counter = Counter(preds)
            predictions.append(counter.most_common(1)[0][0])
        
        return predictions  # Return an array of predictions for all samples
    