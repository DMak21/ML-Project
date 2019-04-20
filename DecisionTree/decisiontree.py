import numpy as np
import pandas as pd

class DecisionNode():
    """
    This class represents a node in the tree 
    attribute: column that is used for splitting
    threshold: value about which the split happens
    value: if it is a leaf node then store the count of true samples 
    true_branch: if the val>=threshold
    false_branch: if the val<threshold
    """
    def __init__(self, attribute=None, threshold=None, value=None, true_branch=None, false_branch=None):
        self.attribute = attribute          
        self.threshold = threshold          
        self.value = value                  
        self.true_branch = true_branch      
        self.false_branch = false_branch 
        
class DecisionTree(object):
    """
    This is the main Decision tree class
    root: the root node of the tree
    min_info_gain: if the information gain made by the split is lesser than this value, the node does not split
    max_depth: the maximum depth of the tree
    split_val_metric: the metrics that can be used are mean and median
    split_node_criterion: the criterion to build the tree, it can be entropy or gini
    """
    def __init__(self, min_info_gain=1e-7, max_depth=float("inf"), split_val_metric = 'mean', split_node_criterion = 'entropy'):
        self.root = None
        self.min_info_gain = min_info_gain
        self.max_depth = max_depth
        self.split_val_metric = split_val_metric
        self.split_node_criterion = split_node_criterion
        self.impurity_calc = None
        self.leaf_val = None
        self.one_hot = None
    
    def fit(self, X, y):
        """
        The function that fits the data to the tree
        """
        self.one_hot = len(np.shape(y)) == 1
        data = np.c_[X, y]
        self.leaf_val = self.count_values
        if self.split_node_criterion.lower() == 'entropy':
            self.impurity_calc = self.info_gain_entropy
        if self.split_node_criterion.lower() == 'gini':
            self.impurity_calc = self.info_gain_gini
        self.root = self.buildTree(X,y)
        
    def divide_on_feature(self, X, attribute, threshold):
        """
        Divide on the particular attribute
        """
        split = None
        if isinstance(threshold, int) or isinstance(threshold, float):
            split = lambda test: test[attribute] >= threshold
        else:
            split = lambda test: test[attribute] == threshold

        left = np.array([sample for sample in X if split(sample)])
        right = np.array([sample for sample in X if not split(sample)])

        return np.array([left, right])
        
    def buildTree(self, X, y, curr_depth = 0):
        """
        Building the tree
        """
        largest_impurity = 0
        best_criteria = None
        best_splits = None
        
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)
            
        data = np.c_[X,y]
        
        n_samples, n_features = np.shape(X)
        
        if curr_depth <= self.max_depth:
            
            for attr in range(n_features):
                
                try:
                    feat_val = np.expand_dims(X[:, attr], axis=1)
                except:
                    feat_val = np.expand_dims(X.iloc[:, attr], axis=1)
                    
                if self.split_val_metric.lower() == 'mean':
                    threshold = np.mean(feat_val)
    
                if self.split_val_metric.lower() == 'median':
                    threshold = np.median(feat_val)
                
                d_1, d_2 = self.divide_on_feature(data, attr, threshold)
                
                if len(d_1)>0 and len(d_2)>0:
                    y_1 = d_1[:, n_features:]
                    y_2 = d_2[:, n_features:]
                    
                    impurity = np.abs(self.impurity_calc(y, y_1, y_2))
                    if impurity > largest_impurity:
                        largest_impurity = impurity
                        best_criteria = {"attr": attr, "threshold": threshold}
                        best_splits = {
                            "leftX" : d_1[:, :n_features],
                            "lefty" : d_1[:, n_features:],
                            "rightX": d_2[:, :n_features],
                            "rightY": d_2[:, n_features:]
                        }
        
        if largest_impurity > self.min_info_gain:
            true_branch = self.buildTree(best_splits["leftX"], best_splits["lefty"], curr_depth + 1)
            false_branch = self.buildTree(best_splits["rightX"], best_splits["rightY"], curr_depth + 1)
            
            return DecisionNode(attribute = best_criteria["attr"], threshold = best_criteria["threshold"], true_branch=true_branch, false_branch=false_branch)
        
        leaf_val = self.leaf_val(y)
        
        return DecisionNode(value = leaf_val)
        
    def predVal(self, X, tree = None):
        """
        The helper function for predicting values
        """
        if tree is None:
            tree = self.root
            
        if tree.value is not None:
#             print(tree.value)
            #     print(i)
            cl = []
            for j in tree.value:
                cl.append(j)
            #     print(cl)
            max_class = None
            max_val = 0
            for j in range(len(tree.value)):
                t = tree.value.get(cl[j])
                #         print(t)
                if t > max_val:
                    max_val = t
                    max_class = cl[j]
#             pre.append(max_class)
            return max_class
        
        feat_val = X[tree.attribute]
        
        branch = tree.false_branch
        
        if isinstance(feat_val, int) or isinstance(feat_val, float):
            if feat_val >= tree.threshold:
                branch = tree.true_branch
                
        elif feat_val == tree.threshold:
            branch = tree.true_branch
            
        return self.predVal(X, branch)

    def predict(self, X):
        """
        This function predicts using the test data
        """
        X = np.array(X)
        pred = [self.predVal(i) for i in X]
        return pred
        
    def print_tree(self, tree=None, indentation=" "):
        """
        Helps print the tree if you need it
        """
        if tree is None:
            tree = self.root
            
        if tree.value is not None:
            print(tree.value)
            
        else:
            print("%s - %s ? " % (tree.attribute, tree.threshold))
            print("%s true =>" % (indentation), end="")
            self.print_tree(tree.true_branch, indentation + indentation)
            print("%s False =>" % (indentation), end ="")
            self.print_tree(tree.false_branch, indentation + indentation)
    
    def entropy(self, rows):
        """
        calculates the entropy used in ID3 
        """
        entropy=0
        count=self.count_values(rows)
        for label in count:
            p=count[label]/float(len(rows))
            entropy-=p*np.log2(p)
        return entropy
    
    def gini(self, rows):
        """
        calculates the gini impurity
        """
        count=self.count_values(rows)
        impurity=1
        for label in count:
            probab_of_label=count[label]/float(len(rows))
            impurity-=probab_of_label**2
        return impurity

    def info_gain_entropy(self, y, y1, y2):
        """
        info gain calculated using the entropy
        """
        p =float(len(y1))/len(y1)+len(y2)
        return self.entropy(y)-p*self.entropy(y1)-(1-p)*self.entropy(y2)
    
    def info_gain_gini(self, y, y1, y2):
        """
        info gain calculated using the gini impurity
        """
        p =float(len(y1))/len(y1)+len(y2)
        return self.gini(y)-p*self.gini(y1)-(1-p)*self.gini(y2)
    
    def count_values(self, rows):
        """
        count the number of occurrences of the predicting class 
        """
        count = {}
        for row in rows: 
            label = row[-1]

            if label not in count: 
                count[label] = 0
            count[label]+=1
        return count