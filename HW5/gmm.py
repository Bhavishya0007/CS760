import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Generating synthetic dataset with same seed for KMEANS and GMM
np.random.seed(10)
n = 100
sigma_values = np.array([0.5, 1, 2, 4, 8])
datasets = []
for sigma in sigma_values:
    P_a = np.random.multivariate_normal([-1, -1], sigma*np.array([[2, 0.5], [0.5, 1]]), n)
    P_b = np.random.multivariate_normal([1, -1], sigma*np.array([[1, -0.5], [-0.5, 2]]), n)
    P_c = np.random.multivariate_normal([0, 1], sigma*np.array([[1, 0], [0, 2]]), n)
    X = np.concatenate((P_a, P_b, P_c), axis=0)
    datasets.append(X)

# Perform K-means clustering and EM algorithm for GMMs on each dataset
k = 3
em_obj_vals = []
em_acc_vals = []
for i in range(np.shape(sigma_values)[0]):
    X = datasets[i]
    
    # EM algorithm for GMMs
    gm = GaussianMixture(n_components=k, random_state=10).fit(X)
    em_obj_vals.append(gm.lower_bound_)
    em_acc = np.sum(np.argmax(gm.predict_proba(X)[:n], axis=1) == 0) + np.sum(np.argmax(gm.predict_proba(X)[n:2*n], axis=1) == 1) + np.sum(np.argmax(gm.predict_proba(X)[2*n:], axis=1) == 2)
    em_acc_vals.append(em_acc / (3*n))

print("gmm obj vals :\n")
print(em_obj_vals)
print("gmm acc vals :\n")
print(em_acc_vals)

# Plot results
plt.figure(0)
plt.plot(sigma_values, em_obj_vals, '-o', label='GMM')
plt.xlabel(r'$\sigma$')
plt.ylabel('Clustering objective')
plt.legend()
plt.title('GMM Clustering Objective vs sigma')
#plt.show()
plt.savefig('CO_gmm.png')
plt.figure(1);
plt.plot(sigma_values, em_acc_vals, '-o', label='GMM')
plt.xlabel(r'$\sigma$')
plt.ylabel('Clustering accuracy')
plt.legend()
plt.title('GMM Clustering Accuracy vs sigma')
plt.savefig('CA_gmm.png')
