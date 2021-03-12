"""

--> DIMENSIONALITY REDUCTION WITH PCA.

*Just an implementation of PCA in Python with Numpy.
*Author: Huy Nguyen

*SOURCE:
    - Matrix rank: https://www.mathsisfun.com/algebra/matrix-rank.html
    - PCA:
        + http://www.sci.utah.edu/~shireen/pdfs/tutorials/Elhabian_PCA09.pdf
        + http://www.stat.columbia.edu/~fwood/Teaching/w4315/Fall2009/pca.pdf

"""

#import matplotlib.pyplot as plt
import numpy as np

def generate_gauss_data(m=1, n=1):
    data = [] # Generate random standard norm data
    for i in range(n):
        data.append(np.random.randn(m))
    return np.array(data).transpose()

def mean_normalization(X):
    for i in range(np.size(X,1)):  # Normalize each feature column.
        X[:,i] = (X[:,i]-X[:,i].mean()) / (X[:,i].max()-X[:,i].min())
    return X

def covariance_matrix(X):  # Data needs to has ~ 0 mean to use this function
    m = np.size(X,0)
    return (1/(m-1)) * (X.transpose()@X)

def pca(X, k, log=False):
    """
    Find the covariance matrix Y that minizes redundancy, measured by covariance matrix. In another word, maximize the variances on the diagonal line, and minimize the covariances off the diagonal line the covariance matrix.
    """

    X = mean_normalization(X)  # Zero mean data.
    covariance_X = covariance_matrix(X)  # Symmetric matrix
    eig_values_X, eig_vectors_X = np.linalg.eig(covariance_X)  # Note that since covariance_X is symmetric, eigen vectors should orthogonal to each other.

    """
    The eigenvector with highest eigenvalue is the principal component of the data set.
    """
    principal_components = list(zip(eig_values_X, eig_vectors_X.transpose()))
    principal_components = sorted(principal_components, key=(lambda x: x[0]), reverse=True)
    principal_components = np.array([i[1] for i in principal_components]).transpose()[:, 0:k]

    """
    OUTPUT DATASET (Reduced Dimentions)
    """
    Y = X@principal_components  # output data.

    if log:
        covariance_Y = covariance_matrix(Y)
        eig_values_Y, eig_vectors_Y = np.linalg.eig(covariance_Y)

        # Calculate explained variance.
        var_retained = (np.sum(eig_values_Y)/np.sum(eig_values_X)) * 100
        print(f"Variance retained: {var_retained:.2f} %")

    return Y

def evaluate_variance_retained(X, Y, log=True):
    return var_retained

# PARAMS
m = 10000  # Num data points.
n = 99  # Num features of dataset.
k = 98 # Num features applied to PCA.

# Generate data.
X = generate_gauss_data(m, n)

# Apply PCA
Y = pca(X, k, log=True)
