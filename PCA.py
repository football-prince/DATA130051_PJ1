'''
Author: Zihao Cheng
Student number: 21307130080
School of data science

Homepage: https://github.com/football-prince/DATA130051_PJ1

This pyhton fuctuin implements a PCA function.
'''

import numpy as np
from matplotlib import pyplot as plt

def manual_PCA(data, n_components=3):
    # Centering the data (subtract the mean of each dimension)
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    
    # Compute the covariance matrix
    covariance_matrix = np.cov(centered_data, rowvar=False)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select the top 'n_components' eigenvectors
    principal_components = sorted_eigenvectors[:, :n_components]
    
    # Transform the data to the new subspace
    transformed_data = np.dot(centered_data, principal_components)
    
    return transformed_data

def nnPCA(nn, affine=False):
    center_x = nn.weights[0].copy()
    PCA_X = manual_PCA(center_x, n_components=3)
    if affine:
        minimum, maximum = np.min(PCA_X), np.max(PCA_X)
        PCA_X = (PCA_X - minimum) * (1. / (maximum - minimum))
    else:
        PCA_X = np.clip(PCA_X, -1., 1.) * .5 + .5
    
    return PCA_X.reshape((28, 28, 3))