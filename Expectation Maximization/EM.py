import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.stats as sp
from IPython import display

class GaussianMixModel(object):
    def __init__(self, X, k):
        
        # Make sure data is in readable format
        X = np.asarray(X)

        # Initialize variables from input set
        self.m, self.n = X.shape
        self.data = X.copy()

        # Initialize Number of Desired Clusters
        self.k = k


    def _init(self):

        # Find mean and std for each dimension in set
        means_ = []
        stds_ = []
        for i in range(self.n):
            means_ += [np.mean(self.data[:,i])]
            stds_ += [np.std(self.data[:,i])]

        # Initialize parameters for EM, according to Example 1.1.1
        self.mean_arr = np.asmatrix(
        np.array(stds_)*np.random.random((self.k,self.n)) + means_)
        self.sigma_arr = np.array([np.std(self.mean_arr, axis = 0) 
        + np.identity(self.n) for i in range(self.k)])
        self.phi = np.ones(self.k)/self.k

        # Initialize Latent Variable giving probability of each point for each distribution
        self.Z = np.asmatrix(np.empty((self.m, self.k), dtype=float))
        


    def e_step(self):

        # Calculate probabilities for each point in each dist.
        # Iterate over whole dataset, and over number of clusters
        for i in range(self.m):
            
            # Initialize denominator counter (for variable Z) 
            den = 0

            for j in range(self.k):

                # Calculate PDF with given parameters and update denom. counter 
                num = sp.multivariate_normal.pdf(self.data[i, :],
                   self.mean_arr[j].A1,
                   self.sigma_arr[j]) *\
                   self.phi[j]
                den += num

                # Calculate latent variable (prob of each point for each dist.)
                self.Z[i, j] = num
            self.Z[i, :] /= den
            assert self.Z[i, :].sum() - 1 < 1e-4  

    def m_step(self):
        # Update means, covariances, cluster probs. for each cluster
        # Iterate over all clusters
        for j in range(self.k):

            # Update Cluster Probabilities
            const = self.Z[:, j].sum()
            self.phi[j] = 1/self.m * const

            # Intiialize temporary variables for updating
            _mu_j = np.zeros(self.n)
            _sigma_j = np.zeros((self.n, self.n))

            # Calculate new mus and sigmas for each point
            for i in range(self.m):
                _mu_j += (self.data[i, :] * self.Z[i, j])
                _sigma_j += self.Z[i, j] * ((self.data[i, :] - self.mean_arr[j, :]).T 
                * (self.data[i, :] - self.mean_arr[j, :]))

            # Update Cluster means and 
            self.mean_arr[j] = _mu_j / const
            self.sigma_arr[j] = (_sigma_j / const)


    def loglikelihood(self):

        # Initialize log L
        logl = 0

        # Iterate over all points in set
        for i in range(self.m):

            # Initialize temporary variable for calculation
            tmp = 0

            # Iterate over every class
            for j in range(self.k):
                
                # Generate PDF with new values for each cluster
                tmp += sp.multivariate_normal.pdf(self.data[i, :],
                self.mean_arr[j, :].A1,self.sigma_arr[j, :]) * self.phi[j]
            
            # Update variable with new logl
            logl += np.log(tmp)
        return logl

        
    def fit(self, tol=0.001):
        self._init()

        # Initialize convergence-checking params. 
        num_iters = 0
        logl = 1
        previous_logl = 0

        # Initialize while loop for convergence check.
        while(logl-previous_logl > tol):

            # Call Functions built above until convergence.
            previous_logl = self.loglikelihood()
            self.e_step()
            self.m_step()
            num_iters += 1
            logl = self.loglikelihood()

        # Print whether and when convergence has been achieved
        print(f'Converged at iteration {num_iters}.')
