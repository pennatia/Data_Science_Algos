# Angelo Pennati - Johns Hopkins Univeristy - Introduction to Machine Learning

# The code below will be implement three different versions of the K Nearest 
# neighbors algorithm: KNN, Edited KNN & Condensed KNN.

# Import necessary packages (pandas and numpy) for data structures and
# manipulation, as well as our data processing pipeline, dubbed "codebase"

import processing_codebase as cb
import pandas as pd
import numpy as np


# Euclidean Distance Implementation ###########################################

# Define a function to calculate the Euclidean between a given test observation
# and a given training observation. This function is written to accept inputs of
# a pandas Series structure. It also takes care not to calculate the distance 
# for the problem's given class label. Please note that the function below is
# recommended for use with data that has been standardized and/or normalized
# for use. 

def euclidean_distance(obs_1,obs_2,class_label):
    
    # Initialize Distance Metric
    distance = 0
    
    # Iterate through each feature of the observation, dropping the class label
    # if needed. 
    for feature in obs_1.index:
        if feature == class_label:
            continue
    
    # Sum the distance of each feature, and finally output  square root of sums.
        else:
            distance += (obs_1[feature] - obs_2[feature]) ** 2
    return distance ** (1/2)



# Mahalanobis Distance Implementation ##########################################

# Define a function to calculate the Mahalanobis distance between a given test 
# observation and a given training observation. This function is written to 
# accept inputs of a pandas Series structure. Please note that the following 
# function is recommended for use when there is an expectation for high 
# correlation between features present within the dataset. In order to calculate
# the covariance matrix we leverage numpy's linear algebra package

def mahalanobis(obs_1,obs_2,class_label):
    
    # Do not consider the class label feature if  present in the observations. 
    # This is done to avoid further aiding our model in decision making and to
    # real-world applications of the probelm at hand. 
    if class_label in obs_1.index:
        obs_1.drop([class_label])
    if class_label in obs_2.index:
        obs_2.drop([class_label])

    # Convert the input to Numpy arrays    
    x = np.array(obs_1)
    y = np.array(obs_2)

    # Convert the distance between the two points and transpose
    dist_t = (x-y).T
    
    # Calcualte Covariance Matrix between the two observations
    cov_mat = np.stack((x, y), axis = 1)
    cov = np.cov(cov_mat)

    # Calculate the inverse of the Covariance Matrix
    cov_inv = np.linalg.pinv(cov)

    # Calculate the distance between the two points w/out transposing
    dist = (x-y)

    # Calculate the mahalanobis distance and return as function output. 
    D_squared = np.dot(np.dot(dist_t,cov_inv),dist)
    return D_squared



# Gaussian Kernel Implementation #################################################

# Define a function that takes in a series of values (distances, in this case) and 
# returns a series of weights, as per the underlying outcoming Gaussian kernel. In
# this case, we also have a tunable parameter "h", which represents the kernel's 
# spread. Mathematically, this is equivalent to the standard deviation of the 
# Gaussian kernel itself. This function will be used in our regression tasks for 
# the following K Nearest Neighbor implementations. 

def GaussianKernel(distances, h):
    
    # Divide the negative of the distances by the total spread (st. dev) chosen
    exp_eq = -(distances)/h

    # Convert to float and exponentiate to return the Gaussian Kernel
    exp_eq = exp_eq.astype(float)
    g = np.exp(exp_eq)

    return g



# Neighbor Identifier Implementation (Helper Function) ############################

# This function has been implemented in order to attain a cleaner implementation of
# all subsequent functions. This function takes in a training set, a testing 
# observation, a number of neighbors "K", a choseb distance metric and a class
# (or predictor variable) name. The function will calculate the distances between 
# the testing observation and each of the training observations inputed. This func-
# tion then sorts the training set by distance, and returns the first "K" neighbors,
# with their full data, as well as with the distance metric. The function was built
# with the intent of being used within our K-Nearest-Neighbor implementation in 
# order to achieve a cleaner and more efficient codebase, such that it may be called
# for each testing observation in subseqeuent function calls. 

def identify_neighbors(train_df,test_obs,k,dist_metric,class_label=None):
    
    # Ask user for Class Label if it is not directly inputted. 
    if class_label == None:
        class_label = input("Please type the class label string.\n")

    # Initialize distances list
    distances = []
    
    # Iterate through the entirety of the training set
    for train_obs in train_df.index:

        # Calculate appropriate distance metric (calling the functions above) to
        # find the distance between the testing observation and each observation in 
        # Training set. Append calculated distance to the distances list. 
        if dist_metric == 'euclidean':
            distance = euclidean_distance(
                train_df.loc[train_obs],test_obs,class_label)
        if dist_metric == 'mahalanobis':
            distance = mahalanobis(train_df.loc[train_obs],test_obs,class_label)
        
        # Append both the distance, as well as all features of the observation list.
        distances.append([train_df.loc[train_obs],distance])

    # Sort the returned distances from lowest to highest        
    dist_array = np.array(distances,dtype=object)
    sorted_dists = dist_array[np.argsort(dist_array[:,1])]

    # Initialize the neighbors output. 
    neighbors = []

    # Iterate through the total number of desired neighbors "K", and append the first
    # "K" neighbors to list. 
    for i in range(k):

        # If there are not enough observations in the training set, retun tuples 0,0
        if i < len(dist_array):
            neighbors.append(sorted_dists[i])
        else:
            neighbors.append([0,0])
    
    # Conver to array and return neighbors as function output
    neighbors = np.array(neighbors)
    return neighbors



# K-Nearest Neighbors Implementation ####################################################

# Here we implement the function that will return the predictions for each training and 
# testing set. Here, we leverage the neighbor identification above (calling it directly)
# to return the nearest neighbors for each test point within the testing dataset accepted
# by the function below. This function indeed accepts a training set, a testing set, a 
# number of neighbors "K", a chosen distance metric, a task (regression/classification),
# and a class label (or predictor). The function then uses the function above to 
# identify the neighbors for each testing observation, and, based on teh task at-hand, 
# returns either the outcome of a plurality vote (classification) or the Gaussian-kernel
# weighted average (regression). The function returns the class/predictor predictions for
# each of the testing points inputted. 

def knn(train_df,test_df,k,dist_metric,task,spread,class_label=None):
    
    # Ask user for class label if it is not directly specified in the function. 
    if class_label == None:
        class_label = input("Please type the class label string.\n")
    
    # Initialize predictions ouput.
    predictions = []

    # Iterate through each observation in the testing dataset. 
    for test_id in test_df.index:

        # Identify neighbors (calling functio) for each of the testing observations. 
        neighbors = identify_neighbors(
            train_df, test_df.loc[test_id], k, dist_metric, class_label)
        
        # Extract only the class labels/predictor from the returned neighbors.
        preds = []
        for i in range(k):
            preds.append(neighbors[i,0][class_label])
        
        # For regression tasks, calculate the predicted value using the Gaussian kernel
        # and a weighted average of the nearest neighbors returned. 
        if task == 'regression':

            # Compute the weights using the Gauassian Kernel
            weights = GaussianKernel(neighbors[:,1],spread)

            # Compute the prediction by performing a weighted average
            predictions.append(np.sum(preds*weights)/len(preds))

        # For classification tasks, employ a plurality vote to determine the class. 
        elif task == 'classification':
            predictions.append(max(set(preds), key = preds.count))

    return predictions



# Correct/Incorrect Classifier (Edited KNN Helper Function) ##############################

# Here, we implement a function to be used within our implementation of the Edited K-NNN 
# algorithm. We use this function to pefrorm K Nearest Neighbors with the newly formed
# training sets continously produced during the Edited KNN training process. The function
# is called within the Edited KNN function until no more correct/incorrect predictions are
# returned. This function hence accepts a "training" and "testing" dataset (which are 
# inherently the same when the function is later called). The function then returns the 
# indices that were correctly and incorrectly classified, such that we may then reduce the 
# training sets in the subsequent function implementation. This function also accepts a 
# tunable "error" parameter to determine whether a given prediction can be considered 
# acceptable for regression tasks. 

def judge_indices(train_df,test_df,k,dist_metric,task,spread,error,class_label):
    
    # Return KNN predictions for the testing set (will be the same as the training set in 
    # the Edited KNN implementation).
    train_prediction = knn(train_df,test_df,k,dist_metric,task,spread,class_label)
    
    # For classifiation tasks, return correct indices and incorrect indices based on an 
    # equality condition. 
    if task == 'classification':
        correct_indices = test_df[test_df[class_label] == train_prediction].index.values
        incorrect_indices = test_df[test_df[class_label] != train_prediction].index.values
    
    # For regression tasks, calculate the acceptable values for each prediction, and create
    # indices where the returned value is within the acceptable boundaries
    elif task == 'regression':
        
        # Calculate acceptable value boundaries (lower and higher)
        lower_b = train_df[class_label] - error
        higher_b = train_df[class_label] + error

        # Initialize outputs
        correct_indices = []
        incorrect_indices =[]
        
        indexer = 0
        
        # Append correct estimations to correct_indices output and append those incorrect
        # to the incorrect indices output. 
        for i in train_df.index:
            if lower_b[i] <= train_prediction[indexer] <= higher_b[i]:
                correct_indices.append(i)
            else:
                incorrect_indices.append(i)
                
            indexer +=1

    # Return the indices that were dubbed correct and incorrect    
    return correct_indices, incorrect_indices



# Edited K-Nearest Neighbor Implementation ####################################################

# This function will accept a training dataset, a testing dataset, a number of neighbors "K", 
# a value for the spreads and acceptable error bound (for regression tasks), a class label, and
# lastly a "method" argument to determine whether to choose whether to only keep the correct or
# incorrect predictions in the reduced training set. This function will leverage the helper 
# function defined above to determine which indices are correct and incorrect. For each method, 
# then, the implementation below extracts a reduced training set with only the deisred indices,
# and repeats the process until no indices are identified as correct/incorrect, based on the
# user's method argument. Once the training set has been thus reduced, the function performs 
# K-Nearest Neighbor estimation on the reduced "new" training set, and on the inputted test set
# Please note that this implementation follows a batch-approach. 

def edited_knn(train_df,test_df,k,dist_metric,task,spread,error,method,class_label=None):
    
    # If class label is not specified, ask user to input it directly. 
    if class_label == None:
        class_label = input("Please type the class label string.\n")
    
    # First pass - retrieve which indices are judged as correct or incorrect in the initially
    # unedited training dataset. 
    correct,incorrect = judge_indices(
        train_df,train_df,k,dist_metric,task,spread,error,class_label)
    
    # Verify which method is being called by the user (Correct, in this case)
    if method == 'C':
        
        # Create a new dataframe with only the observations that were correctly classified.
        new_train_df = train_df.loc[correct]
        
        # Continously perform KNN again on the new training set, and reduce it further until
        # the total number of incorrect estimations equals zero. 
        while len(incorrect) != 0:

            # Judge the indices again (performing KNN) on reduced set
            correct,incorrect = judge_indices(
                new_train_df,new_train_df,k,dist_metric,task,spread,error,class_label)
            
            # Update the new training set
            new_train_df = new_train_df.loc[correct]
    
    # Verify which method is being called by the user (Incorrect, in this case)
    elif method == 'I':
        
        # Create a new dataframe with only observations that were incorrectly classified
        new_train_df = train_df.loc[incorrect]

        # Continously perform KNN again on the new training set, and reduce it further until
        # the total number of correct estimations equals zero. 
        while len(correct) != 0:
            
            # Judge indices again (performing KNN) on reduced set
            correct,incorrect = judge_indices(
                new_train_df,new_train_df,k,dist_metric,task,spread,error,class_label)
            
            # Update the new training set
            new_train_df = new_train_df.loc[incorrect]
    
    # Perform KNN using the reduced training set and the inputed test set, returning preds
    test_predictions = knn(new_train_df,test_df,k,dist_metric,task,spread,class_label)
    return test_predictions



# Condensed K-Nearest Neighbor Implementation ##################################################

# This function accepts a training set, a testing set, a number "K" of neighbors, a distance 
# metric to be used, a spread and error threshold (for regression tasks), and a class label. The
# function then estbalished a shrinking candidate pool of training points (dubbed shrink_df). It
# picks the dataset's first observation, and initialize a new dataset "Z". The nearest neighbor
# to the first value of "Z" is then returned. If the class label / prediction is dubbed to be 
# correct, that value is remvoved from the candidate pool, and the nearest neighbor is 
# returned once more. The process is repeated until an "incorrect" nearest neighbor is returned.
# That point is then appended to "Z", and judged in the same way. The process is repeated until
# Z ceases to grow. The function below then runs K-Nearest-Neighbor estimation using "Z" as a 
# training set, and the inputted test set. 


def condensed_knn(train_df,test_df,k,dist_metric,task,spread,error,class_label=None):
    
    # If class label is not specified, ask user to specifiy it by typing.
    if class_label == None:
        class_label = input("Please type the class label string.\n")

    # Initialize shrinking candidate pool.
    shrink_df = train_df

    # Initialize Z by grabbing the first observation of the dataset.
    z = train_df.drop(train_df.iloc[1:len(shrink_df)].index)

    # Identify the lenght of Z in order to perform while-loop dependent on its size
    index_vals = z.index.tolist()
    
    # Initialize while-loop iterator
    i = 0

    # Repeat the process until Z stops growing (length of indices does not change)
    while i < len(index_vals) :
        
        # Break out of the loop if there are less training observations than desired neighbors
        # for the final output
        if len(shrink_df) <= k:
                break
        
        # Define the current "Z" observation to be estimated.
        train_id = index_vals[i]

        # Find current "Z" observation's nearest neighbor and its class
        neighbors = identify_neighbors(
            shrink_df, z.loc[train_id], k, dist_metric, class_label)
        prediction  = neighbors[0,0][class_label]

        # For classification tasks, continuously drop the nearest neighbors matching the class
        # label from the candidate pool, and repeat the process. 
        if task == 'classification':

            while prediction == z.loc[train_id][class_label]:

                # Drop the observation from the candidate pool
                shrink_df = shrink_df.drop(neighbors[0,0].name)
                
                # If dataframe is shrinked too far, then break out of this loop (as Z can 
                # no longer properly be estimated or grown))
                if len(shrink_df) <= k:
                    break

                # Return the nearest neighbor again, repeating until it is different from 
                # curernt observatrion in "Z"
                neighbors = identify_neighbors(
                    shrink_df, z.loc[train_id], k, dist_metric, class_label)
                prediction = neighbors[0,0][class_label]

            # Once the nearest neighbors is of a different class, append it to Z
            z.loc[neighbors[0,0].name] = neighbors[0,0].values
            
            # Drop the current point from the candidate pool (if while-loop not entered)
            if neighbors[0,0].name in shrink_df.index:
                shrink_df = shrink_df.drop(neighbors[0,0].name)
            
            # Update the index values & iterator
            index_vals = z.index.tolist()
            i += 1
        
        # For regression tasks, define an acceptable range for the value to be considered 
        # correct or incorrect, then repreat the process as above. 
        elif task == 'regression':

            # Find current value and define acceptable range
            nb_val = z.loc[train_id][class_label]
            acceptable_range = [nb_val-(nb_val*error), nb_val+(nb_val*error)]

            # While the nearest neighbor is within the acceptable range, remove it from the
            # candidate pool and repeat the process
            while min(acceptable_range) <= prediction <= max(acceptable_range):
                
                # Drop observation from candidate pool
                shrink_df = shrink_df.drop(neighbors[0,0].name)

                # Break out of while loop if Z can no longer appropriately grow
                if len(shrink_df) <= k:
                    break

                # Perform predictions again using the nearest neighbor. 
                neighbors = identify_neighbors(
                    shrink_df, z.loc[train_id], k, dist_metric, class_label)
                prediction = neighbors[0,0][class_label]

            # Update "Z" when a differing value has been foound
            z.loc[neighbors[0,0].name] = neighbors[0,0].values

            # Drop current point from candidate pool (if while-loop not entered)
            if neighbors[0,0].name in shrink_df.index:
                shrink_df = shrink_df.drop(neighbors[0,0].name)
            
            # Update index values and iterator
            index_vals = z.index.tolist()
            i += 1
    
    # Raise an exception if the returned Dataset "Z" has feweer observations than the K
    # desired for the final estimation. 
    if len(z) < k:
        raise Exception('Under current conditions, Z has fewer observations than K.')

    # Perform estimation using the reduced set "Z" as the training set.  
    test_predictions = knn(z,test_df,k,dist_metric,task,spread,class_label)
    return test_predictions

    
