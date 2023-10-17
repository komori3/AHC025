from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':

    with open('../vs/solver/plot3.txt', 'r', encoding='utf-8') as f:
        lines = [line for line in str(f.read()).split('\n') if not line == '']
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = []
    ys = []
    zs = []
    cs = []
    for line in lines:
        x, y, z, c = map(int, line.split(' '))
        xs.append(x)
        ys.append(y)
        zs.append(z)
        cs.append(c)

    X = np.arange(30000).reshape(10000, 3)
    X[:,0] = xs
    X[:,1] = ys
    X[:,2] = zs
    
    Y = np.array(cs)
    Y -= 1
    
    clf = LogisticRegression().fit(X, Y)

    Z = clf.predict(X)

    print(sum(Y == Z))

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    print(clf.coef_)
    print(clf.intercept_)
    print(f'{clf.coef_[0][0]:.15f} {clf.coef_[0][1]:.15f} {clf.coef_[0][2]:.15f} {clf.intercept_[0]:.15f}')

    score = 0
    for x, y in zip(X, Y):
        z = np.sum(clf.coef_ * x) + clf.intercept_
        z = sigmoid(z)
        v = 0 if z < 0.5 else 1
        if y == v:
            score += 1

    print(score)
