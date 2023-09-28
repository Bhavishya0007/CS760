# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import operator
import json
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

Axes3D = Axes3D


class DecisionTree:
    def __init__(self):
        self.tree = None
        self.features = list
        self.XTrain = np.array
        self.yTrain = np.array
        self.num_feats = int
        self.train_size = int
        self.nodes = 0

    def fit(self, X, y):
        self.XTrain = X
        self.yTrain = y
        self.features = list(X.columns)
        self.train_size = X.shape[0]
        self.num_feats = X.shape[1]

        df = X.copy()
        df['category'] = y.copy()

        self.tree = self.make_subtree(df)
        s = str(self.tree)
        s = s.replace("\'", "\"")
        json_object = json.loads(s)
        printprint(json.dumps(json_object, indent=2))

    def make_subtree(self, df, tree=None):
        feature, cutoff = self.find_best_split(df)
        categories, count = np.unique(df['category'], return_counts=True)
        if tree is None:
            tree = {feature: {}}
        if cutoff == None:
          tree[feature]['>=' + str(-1.5) + ' then'] = 1
          return tree
        
        self.nodes +=1  

        # Left Child
        new_df = self.split_table(df, feature, cutoff, operator.ge)
        categories, count = np.unique(new_df['category'], return_counts=True)

        if len(count) == 1:  # all are same category
            tree[feature]['>=' + str(cutoff) + ' then'] = categories[0]
        else:
            tree[feature]['>=' + str(cutoff) + ' then'] = self.make_subtree(new_df)

        # Right Child
        new_df = self.split_table(df, feature, cutoff, operator.lt)
        categories, count = np.unique(new_df['category'], return_counts=True)

        if len(count) == 1:  # all are same category
            tree[feature]['else ' + '<' + str(cutoff)] = categories[0]
        else:
            tree[feature]['else ' + '<' + str(cutoff)] = self.make_subtree(new_df)

        return tree

    def find_best_split(self, df):
        igr = []
        thresholds = []

        for feature in list(df.columns[:-1]):
            entropy_parent = self.entropy(df)  # H(T)
            info_gain_ratio, threshold = self.entropy_feature(df, feature, entropy_parent)  # H(T|a)

            igr.append(info_gain_ratio)
            thresholds.append(threshold)
         
        #print(thresholds)
        #print(igr) 
        return df.columns[:-1][np.argmax(igr)], thresholds[np.argmax(igr)]

    def entropy(self, df):
        entropy = 0
        for target in np.unique(df['category']):
            probability = df['category'].value_counts()[target] / len(df['category'])
            entropy += -probability * np.log2(probability)

        return entropy

    def entropy_feature(self, df, feature, entropy_parent):
        threshold = None
        info_gain_ratio = 0
        #print("\n Feature: {}".format(feature))

        for pivot in np.unique(df[feature]):
            cur_entropy = 0
            cutoff = pivot
            weightage_entropy = 0
            for operation in [operator.lt, operator.ge]:
                entropy_feature = 0
                den = 0
                for category in np.unique(df['category']):
                    num = len(df[feature][operation(df[feature], cutoff)][df['category'] == category])
                    den = len(df[feature][operation(df[feature], cutoff)])

                    if den == 0:
                        continue
                    probability = num / den
                    if probability > 0:
                        entropy_feature += -probability * np.log2(probability)

                weightage = den / len(df)
                cur_entropy += weightage * entropy_feature
                if weightage > 0:
                    weightage_entropy += -weightage * np.log2(weightage)
            if weightage_entropy == 0:
                print("\n Mutual Information: {}".format(entropy_parent - cur_entropy))
                print("If Split at: {}".format(cutoff))
                continue
            cur_info_gain_ratio = (entropy_parent - cur_entropy) / weightage_entropy
            print("\n Info Gain: {}".format(cur_info_gain_ratio))
            print("If Split at: {}".format(cutoff))

                
            if cur_info_gain_ratio > info_gain_ratio:
                info_gain_ratio = cur_info_gain_ratio
                threshold = cutoff

        return info_gain_ratio, threshold

    def split_table(self, df, feature, pivot, operation):
        return df[operation(df[feature], pivot)].reset_index(drop=True)

    def predict(self, X):
        results = []
        lookup = {key: i for i, key in enumerate(list(X.columns))}

        for i in range(len(X)):
            results.append(self.predict_X(lookup, X.iloc[i], self.tree))

        return np.array(results)

    def predict_X(self, lookup, x, tree):
        for node in tree.keys():
            val = x[node]
            cutoff = str(list(tree[node].keys())[0]).split('>=')[1].split(' ')[0]

            if val >= float(cutoff):  # Left Child
                tree = tree[node]['>=' + cutoff + ' then']
            else:  # Right Child
                tree = tree[node]['else ' + '<' + cutoff]

            prediction = str

            if type(tree) is dict:
                prediction = self.predict_X(lookup, x, tree)
            else:
                predict = tree
                return predict

        return prediction


def error_score(ytrue, ypred):
    return round(float(sum(ypred != ytrue)) / float(len(ytrue)) * 100, 2)


if __name__ == '__main__':
    print('\n-------------- Q2.1-Q2.6------------------------')

    data = pd.read_table('data/D1.txt', sep=" ", header=None, names=["X1", "X2", "Y"])
    Xtest=pd.DataFrame(np.random.uniform(0,1,size=(1000000,2)), columns=list(['X1','X2']))

    # Split Features and target
    X, y = data.drop([data.columns[-1]], axis=1), data[data.columns[-1]]
    

    dt_clf = DecisionTree()
    dt_clf.fit(X, y)
    #print("\nTrain Error: {}".format(error_score(y, dt_clf.predict(X))))
    print(dt_clf.nodes)

    plt.scatter(data['X1'], data['X2'], c=data['Y'])
    plt.show()
    plt.scatter(Xtest['X1'], Xtest['X2'], c=dt_clf.predict(Xtest))
    plt.show()