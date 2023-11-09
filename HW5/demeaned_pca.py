import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# defining PCA 
def pca(X, d):
    X_centered = X
    X_mean = np.mean(X_centered, axis=0)
    # Compute SVD
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
    X_reconstructed += X_mean
    return X_pca, params, X_reconstructed, s
# Generate data
X = np.genfromtxt('data/data2D.csv', delimiter=',')
X_mean = np.mean(X, axis=0)
# Demeaned PCA
X_centered = X - X_mean
# Run PCA with 2 dimensions
X_pca, params, X_reconstructed , singular_values = pca(X_centered, 1)
X_reconstruction_error = np.square(np.linalg.norm(X_centered-X_reconstructed))
X_reconstruction_error = X_reconstruction_error/X_centered.shape[0]
print("Reconstruction Error Demeaned PCA : ", X_reconstruction_error)
plt.xlabel("x1 : Feature 1")
plt.ylabel("x2 : Feature 2")
plt.title("Demeaned PCA : Scatter Plot")
plt.scatter(X_centered[:,0], X_centered[:,1], c="orange")
plt.scatter(X_reconstructed[:,0], X_reconstructed[:,1], c="blue")
plt.legend(["original data", "reconstructed data"])
plt.savefig('demeaned_pca.png')
#plt.show()