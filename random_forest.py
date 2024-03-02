# Assignmnet 1
# Liam Duncan - 301476562
# February 16, 2024


from dataclasses import dataclass, replace
from posixpath import split
from typing import Optional, Sequence, Mapping
import numpy as np
import pandas as pd
import random


class Node(object):
    def __init__(self, node_size: int, node_class: str, depth: int, single_class:bool = False):
        # Every node is a leaf unless you set its 'children'
        self.is_leaf = True
        # Each 'decision node' has a name. It should be the feature name
        self.name = None
        # All children of a 'decision node'. Note that only decision nodes have children
        self.children = {}
        # Whether corresponding feature of this node is numerical or not. Only for decision nodes.
        self.is_numerical = None
        # Threshold value for numerical decision nodes. If the value of a specific data is greater than this threshold,
        # it falls under the 'ge' child. Other than that it goes under 'l'. Please check the implementation of
        # get_child_node for a better understanding.
        self.threshold = None
        # The class of a node. It determines the class of the data in this node. In this assignment it should be set as
        # the mode of the classes of data in this node.
        self.node_class = node_class
        # Number of data samples in this node
        self.size = node_size
        # Depth of a node
        self.depth = depth
        # Boolean variable indicating if all the data of this node belongs to only one class. This is condition that you
        # want to be aware of so you stop expanding the tree.
        self.single_class = single_class

    def set_children(self, children):
        self.is_leaf = False
        self.children = children

    def get_child_node(self, feature_value)-> 'Node':
        if not self.is_numerical:
            return self.children[feature_value]
        else:
            if feature_value >= self.threshold:
                return self.children['ge'] # ge stands for greater equal
            else:
                return self.children['l'] # l stands for less than


class RandomForest(object):
    def __init__(self, n_classifiers: int,
                 criterion: Optional['str'] = 'gini',
                 max_depth: Optional[int] = None,
                 min_samples_split: Optional[int] = None,
                 max_features: Optional[int] = None):
        """
        :param n_classifiers:
            number of trees to generated in the forrest
        :param criterion:
            The function to measure the quality of a split. Supported criteria are “gini” for the Gini
            impurity and “entropy” for the information gain.
        :param max_depth:
            The maximum depth of the trees.
        :param min_samples_split:
            The minimum number of samples required to be at a leaf node
        :param max_features:
            The number of features to consider for each tree.
        """
        self.n_classifiers = n_classifiers
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.criterion_func = self.entropy if criterion == 'entropy' else self.gini


    def fit(self, X: pd.DataFrame, y_col: str)->float:
        """
        :param X: data
        :param y_col: label column in X
        :return: accuracy of training dataset
        """
        features = self.process_features(X, y_col)
        i = 0
        while i < self.n_classifiers:
            random_feature_indices = np.random.choice(len(features), size=self.max_features, replace=False)
            random_features = [features[j] for j in random_feature_indices]
            random_data_indices = np.random.choice(len(X), size=len(X), replace=True)
            
      
            self.trees.append(self.generate_tree(X.iloc[random_data_indices], y_col, random_features))
            i += 1
            print("tree ", i, "completed")
        
        return self.evaluate(X, y_col)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts the class labels for the given dataset using the trained random forest model.

        :param X: The input data samples.
        :return: Aggregated predictions of all trees on X using voting mechanism.
        """
        predictions = []
        i = 0
        for tree in self.trees:
            tree_preds = []
            for _, sample in X.iterrows():
                node = tree
                i +=1
                while not node.is_leaf:
                    
                    feature_value = sample[node.name]
                    if node.is_numerical:  
                        
                        if feature_value < node.threshold:
                            node = node.children['l']  
                        else:
                            node = node.children['ge']  
                    else: 
                        
                        if feature_value in node.children: 
                            node = node.children[feature_value]
                        else:
                            node = node
                            break
                if isinstance(node.node_class, pd.Series):
                    tree_preds.append(node.node_class.iloc[0])
                else:
                    tree_preds.append(node.node_class)

            predictions.append(tree_preds)  

        aggregated_predictions = []
        for sample_predictions in zip(*predictions):  
            class_counts = {}
            for pred in sample_predictions:
                if pred in class_counts:
                    class_counts[pred] += 1
                else:
                    class_counts[pred] = 1

            majority_class = max(class_counts, key=class_counts.get)
            aggregated_predictions.append(majority_class)

        return np.array(aggregated_predictions)


    def evaluate(self, X: pd.DataFrame, y_col: str)-> int:
        """
        :param X: data
        :param y_col: label column in X
        :return: accuracy of predictions on X
        """
        preds = self.predict(X)
        acc = sum(preds == X[y_col]) / len(preds)
        return acc

    def generate_tree(self, X: pd.DataFrame, y_col: str,   features: Sequence[Mapping])->Node:
        """
        Method to generate a decision tree. This method uses self.split_tree() method to split a node.
        :param X:
        :param y_col:
        :param features:
        :return: root of the tree
        """
        root = Node(X.shape[0], X[y_col].mode(), 0)
        self.split_node(root, X, y_col, features)
        return root

    def split_node(self, node: Node, X: pd.DataFrame, y_col:str, features: Sequence[Mapping]) -> None:
        """
        This is probably the most important function you will implement. This function takes a node, uses criterion to
        find the best feature to split it, and splits it into child nodes. I recommend to use recursive programming to
        implement this function but you are of course free to take any programming approach you want to implement it.
        :param node:
        :param X:
        :param y_col:
        :param features:
        :return:

        """
        best_split = None
        is_numeric = False
        best_crit = float('inf')
        if (len(X) > self.min_samples_split) and node.depth < self.max_depth:
            for feature_map in features:
                feature_name = feature_map['name']
                
                if feature_map['dtype'] == int:
                        mean = X[feature_name].mean()
                        new_left_df = X[X[feature_name] < mean].copy()
                        new_right_df = X[X[feature_name] >= mean].copy()

                        
                        if len(new_left_df) == 0 or len(new_right_df) == 0:
                            continue  
                        
                        total_samples = len(new_left_df) + len(new_right_df)
                        left_gini = self.criterion_func(new_left_df, feature_name, y_col)
                        right_gini = self.criterion_func(new_right_df, feature_name, y_col)
                        weighted_crit = (len(new_left_df) / total_samples) * left_gini + (len(new_right_df) / total_samples) * right_gini
                        if self.criterion != 'gini':
                            before  = self.criterion_func(X, feature_name, y_col)
                            weighted_crit = before - weighted_crit
                        if weighted_crit < best_crit:
                            best_crit = weighted_crit
                            best_split = {'feature': feature_name, 'value': mean}
                            is_numeric = True
                            
                else:
                    weighted_crit = 0.0
                    total_samples = len(X)
                    unique_values = X[feature_name].unique()
                    for value in unique_values:

                        new_df = X[X[feature_name] == value].copy()
                        subset_crit = self.criterion_func(new_df, feature_name, y_col)
                        subset_size = len(new_df)
                        weighted_crit += (subset_size / total_samples) * subset_crit
                        
                    if self.criterion != 'gini':
                        before  = self.criterion_func(X, feature_name, y_col)
                        weighted_crit = before - weighted_crit
                    if weighted_crit < best_crit:
                        best_crit = weighted_crit
                        best_split = {'feature': feature_name}
                        is_numeric = False                       
                
                        

        
            if best_split:
                feature = best_split['feature']

                if is_numeric == True:
                    mean = X[feature].mean()

                    new_left_df = X[X[feature] < mean].copy()
                    new_right_df = X[X[feature] >= mean].copy()
                    left_node = Node(new_left_df.shape[0], new_left_df[y_col].mode(), node.depth + 1)
                    right_node = Node(new_right_df.shape[0], new_right_df[y_col].mode(), node.depth + 1)

                    node.is_numerical = True
                    node.is_leaf = False
                    node.children['l'] = left_node
                    node.children['ge'] = right_node
                    node.threshold = mean
                    node.name = feature
                    self.split_node(left_node, new_left_df, y_col, [f for f in features if f['name'] != feature])
                    self.split_node(right_node, new_right_df, y_col, [f for f in features if f['name'] != feature])

                else:
                    unique_values = X[feature].unique()
                    node.name = feature
                    for value in unique_values:
                        new_df = X[X[feature] == value].copy()
                        new_node = Node(new_df.shape[0], new_df[y_col].mode(), node.depth + 1)
                        node.children[str(value)] = new_node
                        node.is_leaf = False
                        self.split_node(new_node, new_df, y_col, [f for f in features if f['name'] != feature])        
        


    def gini(self, X: pd.DataFrame, feature: Mapping, y_col: str) -> float:
        """
        Returns gini index of the give feature
        :param X: data
        :param feature: the feature you want to use to get compute gini score
        :param y_col: name of the label column in X
        :return:
        """
        total_size = len(X)
        gini_index = 0.0
        prob_greater = len(X[X[y_col] == '>50K']) / total_size
        prob_less = len(X[X[y_col] == '<=50K']) / total_size
        gini_index = 1 - (prob_greater**2 + prob_less**2)
    
        return gini_index
        

    def entropy(self, X: pd.DataFrame, feature: Mapping, y_col: str) -> float:
        """
        Returns entropy of the given feature
        :param X: data
        :param feature: the feature you want to use to compute entropy
        :param y_col: name of the label column in X
        :return: entropy
        """
        total_size = len(X)
        entropy = 0

        prob_greater = len(X[X[y_col] == '>50K']) / total_size
        prob_less = len(X[X[y_col] == '<=50K']) / total_size

        if prob_greater != 0:
            entropy -= prob_greater * np.log2(prob_greater)
        if prob_less != 0:
            entropy -= prob_less * np.log2(prob_less)

        return entropy

    def process_features(self, X: pd.DataFrame, y_col: str)->Sequence[Mapping]:
        """
        :param X: data
        :param y_col: name of the label column in X
        :return:
        """
        features = []
        for n,t in X.dtypes.items():
            if n == y_col:
                continue
            f = {'name': n, 'dtype': t}
            features.append(f)
        return features