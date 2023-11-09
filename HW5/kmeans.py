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
kmeans_obj_vals = []
kmeans_acc_vals = []
for i in range(np.shape(sigma_values)[0]):
    X = datasets[i]
    
    # K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=10).fit(X)
    kmeans_obj_vals.append(kmeans.inertia_)
    kmeans_acc = np.sum(kmeans.labels_[:n] == 0) + np.sum(kmeans.labels_[n:2*n] == 1) + np.sum(kmeans.labels_[2*n:] == 2)
    kmeans_acc_vals.append(kmeans_acc / (3*n))
    
print("kmeans obj vals :\n")
print(kmeans_obj_vals)
print("kmeans acc vals :\n")
print(kmeans_acc_vals)

# Plot results
plt.figure(0)
plt.plot(sigma_values, kmeans_obj_vals, '-o', label='K-means')
plt.xlabel(r'$\sigma$')
plt.ylabel('Clustering objective')
plt.legend()
plt.title('KMEANS Clustering Objective vs sigma')

#plt.show()
plt.savefig('CO_kmeans.png')
plt.figure(1);
plt.plot(sigma_values, kmeans_acc_vals, '-o', label='K-means')
plt.xlabel(r'$\sigma$')
plt.ylabel('Clustering accuracy')
plt.legend()
plt.title('KMEANS Clustering Accuracy vs sigma')
plt.savefig('CA_kmeans.png')

