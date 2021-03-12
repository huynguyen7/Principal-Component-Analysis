"""

--> DIMENSIONALITY REDUCTION WITH PCA.

*Just an implementation of PCA in Python with Numpy.

*SOURCE:
    - Matrix rank: https://www.mathsisfun.com/algebra/matrix-rank.html
    - PCA:
        + http://www.sci.utah.edu/~shireen/pdfs/tutorials/Elhabian_PCA09.pdf
        + http://www.stat.columbia.edu/~fwood/Teaching/w4315/Fall2009/pca.pdf

"""

import matplotlib.pyplot as plt
import numpy as np

def generate_gauss_data(m=100, n=2, mu=0.0, sigma=2.0):
    data = [] 
    for i in range(n):
        data.append(np.random.randn(m)*sigma+mu)
    return np.array(data).transpose()

def generate_uniform_data(m=100, n=2, low=0.0, high=1.0):
    data = [] 
    for i in range(n):
        data.append(np.random.uniform(low, high, size=m))
    return np.array(data).transpose()

def mean_normalization(X):
    for i in range(np.size(X,1)):  # Normalize each feature column.
        X[:,i] = (X[:,i]-X[:,i].mean()) / (X[:,i].std())
    return X

def covariance_matrix(X):  # Data needs to has ~ 0 mean to use this function
    m = np.size(X,0)
    return (1/(m-1)) * (X.transpose()@X)

def pca(X, k, log=False):
    """
    Find the covariance matrix Y that minimizes redundancy, measured by covariance matrix. In another word, maximize the variances on the diagonal line, and minimize the covariances off the diagonal line the covariance matrix.
    """

    X = mean_normalization(X)  # Zero mean data.
    covariance_X = covariance_matrix(X)  # Symmetric matrix
    eig_values_X, eig_vectors_X = np.linalg.eig(covariance_X)  # Note that since covariance_X is symmetric, eigen vectors should be orthogonal to each others.

    """
    The eigenvector with highest eigenvalue (highest variance) is the principal component of the data set.
    """
    principal_components = list(zip(eig_values_X, eig_vectors_X.transpose()))
    principal_components = sorted(principal_components, key=(lambda x: x[0]), reverse=True)
    principal_components = np.array([i[1] for i in principal_components]).transpose()[:, 0:k]

    """
    OUTPUT DATASET (Reduced Dimensions)
    """
    Y = X@principal_components  # output data.

    if log:
        covariance_Y = covariance_matrix(Y)
        eig_values_Y, eig_vectors_Y = np.linalg.eig(covariance_Y)

        # Calculate explained variance.
        var_retained = (np.sum(eig_values_Y)/np.sum(eig_values_X)) * 100
        print(f"Variance retained: {var_retained:.2f} %")

    return Y

# PARAMS
k = 90 # Num features applied to PCA.

""" Generate data. """
#X = generate_gauss_data(
#        m=10000,  # Num data points.
#        n=100,  # Num features of dataset.
#        mu=0.0,
#        variance=2.0)

X = generate_uniform_data(
        m=10000,
        n=100,
        low=0.0,
        high=1.0)

# Apply PCA
Y = pca(X, k, log=True)
