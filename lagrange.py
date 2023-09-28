import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats

theta = 0
eps = np.finfo(float).eps


def lagrangify(X, Y):
    def ansf(x):
        ans = 0
        for xj, yj in zip(X, Y):
            l = 1
            for xi, yi in zip(X, Y):
                if xj != xi:
                    l *= (x - xi) / (xj - xi)
            ans += l * yj

        return ans

    return ansf


def gaussian(mu, sigma):
    lower, upper = 0, 4 * math.pi
    Xtrain = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(100)
    Ytrain = np.cos(Xtrain + (theta * math.pi) / (10 + eps))
    f_model = lagrangify(Xtrain, Ytrain)
    Xtest = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(100)
    Ytest = np.cos(Xtest + (theta * math.pi) / (10 + eps))

    MSE = np.square(np.subtract(Ytrain, f_model(Xtrain))).mean()
    rsme = math.sqrt(MSE)
    print("Gaussian train RSME error: {}".format(rsme))

    MSE = np.square(np.subtract(Ytest, f_model(Xtest))).mean()

    rsme = math.sqrt(MSE)
    plt.scatter(Xtest, Ytest)
    plt.scatter(Xtest, f_model(Xtest))
    #plt.show()
    print("Sigma: {}".format(sigma))
    print("Gaussian test RSME error: {}".format(rsme))

    return rsme


if __name__ == '__main__':
    print('\n--------------Start of Q4------------------------')
    plt.xlim(0, 4 * math.pi)
    plt.ylim(-1, 1)
    Xtrain = np.random.uniform(0, 4*math.pi, size=100)
    Ytrain = np.cos(Xtrain + (theta * math.pi) / (10 + eps))
    Xtest = np.random.uniform(0, math.pi, size=100)
    Ytest = np.cos(Xtest + (theta * math.pi) / (10 + eps))
    f_model = lagrangify(Xtrain, Ytrain)

    MSE = np.square(np.subtract(Ytrain, f_model(Xtrain))).mean()

    rsme = math.sqrt(MSE)
    print("Uniform train RSME error: {}".format(rsme))

    MSE = np.square(np.subtract(Ytest, f_model(Xtest))).mean()

    rsme = math.sqrt(MSE)
    print("Uniform test RSME error: {}".format(rsme))
    plt.xlim(0, 4 * math.pi)
    plt.ylim(-1, 1)
    plt.scatter(Xtest, Ytest)
    plt.scatter(Xtest, f_model(Xtest))
    #plt.show()
    print(gaussian(2*math.pi-(theta*math.pi)/10, math.pi/6))
    print(gaussian(2 * math.pi - (theta * math.pi) / 10, math.pi / 4))
    print(gaussian(2 * math.pi - (theta * math.pi) / 10, math.pi / 2))




