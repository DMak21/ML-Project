import numpy as np
import pandas as pd
import multiprocessing as mp
from decisiontree import DecisionTree

class RandomForest():
    def __init__(self, n_trees = 100, max_features = None, split_val_metric = 'mean', split_node_criterion = 'gini', 
                 min_gain = 1e-7, max_depth = float("inf"), bootstrap = False, n_jobs = 1):
        self.n_trees = n_trees
        self.max_features = max_features
        self.min_gain = min_gain
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.split_val_metric = split_val_metric
        self.split_node_criterion = split_node_criterion
        self.trees = []
        for _ in range(n_trees):
            self.trees.append(
                DecisionTree(
                    min_info_gain = self.min_gain, 
                    max_depth = self.max_depth,
                    split_val_metric = self.split_val_metric,
                    split_node_criterion = self.split_node_criterion
                ))
            
    def fit(self, X, y):
        y = np.array(y)
        n_feat = np.shape(X)[1]
        if not self.max_features:
            self.max_features = int(np.sqrt(n_feat))
            
        if self.bootstrap == False:
            for i in range(self.n_trees):
                self.trees[i].fit(X, y)
        
        else: 
            subs = self.get_random_subsets(X, y, self.n_trees)
            for i in range(self.n_trees):
#                 print(subs)
                X_sub, y_sub = subs[i]
                idx = np.random.choice(range(n_feat), size=self.max_features, replace=True)
                self.trees[i].feat_i = idx
                X_sub = X_sub[:, idx]
                self.trees[i].fit(X_sub,y_sub)

    def predict(self, X):
        y_preds = np.empty((X.shape[0],len(self.trees)))
        for i, tree in enumerate(self.trees):
            prediction = tree.predict(X)
#             print(prediction)
            if self.bootstrap == True:
                idx = tree.feat_i
                try:
                    prediction = tree.predict(X[:, idx])
                except:
                    prediction = tree.predict(X.iloc[:, idx])
                y_preds[:, i] = prediction
            else: 
                y_preds[:, i] = prediction
        y_pred = []
        for sample_predictions in y_preds:
            y_pred.append(np.bincount(sample_predictions.astype('int')).argmax())
        return y_pred
        
        
    def get_random_subsets(self, X, y, n_sub):
        n_samples = np.shape(X)[0]
        data = np.c_[X, y]
        np.random.shuffle(data)
        subsets = []
        sub_size = int(n_samples // 2)
        for _ in range(n_sub):
            idx = np.random.choice(
                range(n_samples),
                size=np.shape(range(sub_size)))
            X = data[idx][:, :-1]
            y = data[idx][:, -1]
            subsets.append([X, y])
        return subsets