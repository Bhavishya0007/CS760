import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def pca(X, d):
    X_centered = X
    X_mean = np.mean(X_centered, axis=0)
    # Compute SVD of X
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Compute d-dimensional representation of X
    X_pca = X_centered.dot(Vt[:d].T)

    # Compute estimated parameters
    explained_variances = (s ** 2) / (X.shape[0] - 1)
    total_variance = np.sum(explained_variances)
    explained_variance_ratios = explained_variances / total_variance
    params = {'U': U,'s': s,'Vt': Vt,'explained_variances': explained_variances,'explained_variance_ratios': explained_variance_ratios,'X_mean': X_mean}
    # Compute reconstructions
    X_reconstructed = X_pca.dot(Vt[:d])
    return X_pca, params, X_reconstructed, s
# Generate data
X = np.genfromtxt('data/data1000D.csv', delimiter=',')
X_mean = np.mean(X, axis=0)
# Buggy PCA
X_centered = X
# Run PCA with 2 dimension
X_pca, params, X_reconstructed , singular_values = pca(X_centered, 5)
s_log = np.log(singular_values)
plt.grid(alpha=0.5)
plt.ylabel("log(singular_values)")
plt.xlabel("Number of features")
plt.title("KNEE PLOT")
plt.xlim([-1, 200])
plt.plot(s_log)
plt.savefig('knee_plot.png')