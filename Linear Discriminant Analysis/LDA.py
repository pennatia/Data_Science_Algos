import numpy as np
import pandas as pd
from sklearn import preprocessing

class LDA:

    # Define function for data preparation
    def prep_data(cols, dataset, target_label):

        # Pre Process Labels and create numerical vectors
        temp_lbls = preprocessing.LabelEncoder()
        labels = temp_lbls.fit_transform(dataset[target_label])
        Data = dataset.drop([target_label], axis=1)
        if len(cols) > 0:
            Data = Data[cols]
        return Data.values, labels

    # Define function for LDA fitting
    def fit(self, X, y):
        target_classes = np.unique(y)
        mean_vectors = []
        
        # Calculate Mean Vectors & Reshape
        for cls in target_classes:
            mean_vectors.append(np.mean(X[y == cls], axis=0))
        data_mean = np.mean(X, axis=0).reshape(1, X.shape[1])

        # Initialize and Calculate Between class scatter
        B = np.zeros((X.shape[1], X.shape[1]))
        for i, mean_vec in enumerate(mean_vectors):
            n = X[y == i].shape[0]
            mean_vec = mean_vec.reshape(1, X.shape[1])
            mu1_mu2 = mean_vec - data_mean
            B += n * np.dot(mu1_mu2.T, mu1_mu2)
        s_matrix = []

        # Initialize and Calculate within-class scatter
        for cls, mean in enumerate(mean_vectors):
            Si = np.zeros((X.shape[1], X.shape[1]))
            for row in X[y == cls]:
                t = (row - mean).reshape(1, X.shape[1])
                Si += np.dot(t.T, t)
            s_matrix.append(Si)

        # Reshape for following operation
        S = np.zeros((X.shape[1], X.shape[1]))
        for s_i in s_matrix:
            S += s_i
        
        # Perform inverse, matmul and eigen decomposition
        S_inv = np.linalg.inv(S)
        S_inv_B = S_inv.dot(B)
        eig_vals, eig_vecs = np.linalg.eig(S_inv_B)
        idx = eig_vals.argsort()[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]
        return eig_vecs, eig_vals