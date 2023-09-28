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


def error_score(ytrue, ypred):
    return round(float(sum(ypred != ytrue)) / float(len(ytrue)) * 100, 2)


if __name__ == '__main__':
    print('\n--------------Q1 using sklearn------------------------')

    data = pd.read_table('data/Dbig.txt', sep=" ", header=None, names=["X1", "X2", "Y"])

    # Split Features and target
    X, y = data.drop([data.columns[-1]], axis=1), data[data.columns[-1]]

    # text_representation = export_text(dt_clf)
    # print(text_representation)

    train = data.sample(frac=0.8192, random_state=0)
    test = data.drop(train.index)

    Xcord = []
    train32 = train.iloc[0:32, :]
    print(len(train32))

    X, y = train32.drop([train32.columns[-1]], axis=1), train32[train32.columns[-1]]
    # dt_clf = DecisionTree()
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X, y)
    X, y = test.drop([test.columns[-1]], axis=1), test[test.columns[-1]]
    E32 = error_score(y, dt_clf.predict(X))
    print("\nTest Error: {}".format(E32))
    Xcord.append(dt_clf.tree_.node_count)

    train128 = train.iloc[32:160, :]
    print(len(train128))
    X, y = train128.drop([train128.columns[-1]], axis=1), train128[train128.columns[-1]]
    # dt_clf = DecisionTree()
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X, y)
    X, y = test.drop([test.columns[-1]], axis=1), test[test.columns[-1]]
    E128 = error_score(y, dt_clf.predict(X))
    print("\nTest Error: {}".format(E128))
    Xcord.append(dt_clf.tree_.node_count)

    train512 = train.iloc[160:672, :]
    print(len(train512))
    X, y = train512.drop([train512.columns[-1]], axis=1), train512[train512.columns[-1]]
    # dt_clf = DecisionTree()
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X, y)
    X, y = test.drop([test.columns[-1]], axis=1), test[test.columns[-1]]
    E512 = error_score(y, dt_clf.predict(X))
    print("\nTest Error: {}".format(E512))
    Xcord.append(dt_clf.tree_.node_count)

    train2048 = train.iloc[672:2720, :]
    print(len(train2048))
    X, y = train2048.drop([train2048.columns[-1]], axis=1), train2048[train2048.columns[-1]]
    # dt_clf = DecisionTree()
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X, y)
    X, y = test.drop([test.columns[-1]], axis=1), test[test.columns[-1]]
    E2048 = error_score(y, dt_clf.predict(X))
    print("\nTest Error: {}".format(E2048))
    Xcord.append(dt_clf.tree_.node_count)

    X, y = train.drop([train.columns[-1]], axis=1), train[train.columns[-1]]
    # dt_clf = DecisionTree()
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X, y)
    X, y = test.drop([test.columns[-1]], axis=1), test[test.columns[-1]]
    E8192 = error_score(y, dt_clf.predict(X))
    print("\nTest Error: {}".format(E8192))
    Xcord.append(dt_clf.tree_.node_count)

    Xcord = np.array(Xcord)
    Ycord = np.array([E32, E128, E512, E2048, E8192])

    print("List of number of nodes:")
    print(Xcord)

    plt.plot(Xcord, Ycord)
    plt.show()


