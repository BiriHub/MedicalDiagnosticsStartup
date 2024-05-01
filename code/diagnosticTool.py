###
# Code for the diagnostic tool 2° practice part of the assignment , exercise 7.c
# Author: Birindelli Leonardo 
###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

seed=1917953388 #seed
np.random.seed(seed) #used for reproducibility in bootstrap analysis


#Linear Congruential Generator (LCG) function 
def lcg(seed, a, c, m, n, rescale):
    x = seed
    randNums= []
    for i in range(0,n):
        x = (a * x + c) % m
        if(rescale):
            x /= m
        randNums.append(x)
    return randNums


#File name
file_path = 'diabetes.csv'

# Load the 'diabetes.csv' dataset
data = pd.read_csv(file_path)

number_of_rows = len(data) #768
test_set_size = int(0.1*number_of_rows) #76

#Parameters for the LCG
a=3
c=1
m=number_of_rows

#Calculate the indices for the test set (10% of the data)
test_indexes = lcg(seed,a,c,number_of_rows,number_of_rows,False)

test_indices = list(set(test_indexes))  # Ensure uniqueness of indexes

#control if the test set size is correct
if len(test_indices) > test_set_size:
    test_indices = test_indices[:test_set_size]  # Trim to the desired size


#(i.) Split the data into two arrays based on indices (training and test sets)
    
test_data = data.iloc[test_indices] #test set : 10% of the data
training_data = data.drop(test_indices) #training set : 90% of the data

# Prepare the data for logistic regression
# Extract features and target variables for training and testing sets
X_train = training_data.drop('Outcome', axis=1).values
y_train = training_data['Outcome'].values

X_test = test_data.drop('Outcome', axis=1).values
y_test = test_data['Outcome'].values

# Adding intercept term to both X_train and X_test (beta_0)
n_train = X_train.shape[0]
X_train = np.c_[np.ones(n_train), X_train]

n_test = X_test.shape[0]
X_test = np.c_[np.ones(n_test), X_test]

#Function definitions for the logistic regression

# Logistic function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Predict function
def predict(X, beta):
    return sigmoid(np.dot(X, beta))

#Gradient of the loss function
def log_loss_grad(y, X, beta):
    predictions = predict(X, beta)
    return np.dot(X.T,predictions - y) / len(y)

# Gradient Descent
def gradient_descent(X, y, alpha, epsilon, max_iter):
    columns=X.shape[1];
    beta = np.zeros(columns)
    
    i = 0
    difference = 99999.9

    while i < max_iter and difference > epsilon:
        grad = log_loss_grad(y, X, beta)
        new_beta = beta - alpha * grad
        difference = np.linalg.norm(new_beta - beta)
        
        beta = new_beta
        i += 1

    return beta

#(ii.)

# Parameters
alpha = 0.01   # Learning rate
epsilon = 1e-4 # Convergence threshold
max_iter = 10000 # Maximum number of iterations

beta = gradient_descent(X_train, y_train, alpha, epsilon, max_iter)

# Predict on the test set
y_pred = predict(X_test, beta) >= 0.5

# Evaluating the model
accuracy = np.mean(y_pred == y_test)

print("Beta coefficients:", beta)
print("Accuracy on the test set:", accuracy)


# (iv.) Boostrap analysis

print("\nBootstrap analysis has been started! , printing the distributions can take a while...\n")

# Bootstrap function for beta coefficients
def bootstrap_beta(X, y, beta, B):
    n = len(y)
    p = len(beta)
    bootstrap_betas = np.zeros((B, p))
    for i in range(B):
        indices = np.random.choice(range(n), n, replace=True)
        X_resampled = X[indices]
        y_resampled = y[indices]
        resampled_beta = gradient_descent(X_resampled, y_resampled, alpha, epsilon, max_iter)
        bootstrap_betas[i, :] = resampled_beta
    return bootstrap_betas

# Number of bootstrap iterations
B = 1000
original_beta = beta

# Perform bootstrap
bootstrap_betas = bootstrap_beta(X_train, y_train, original_beta, B)

variables = ['Intercept term','Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
significant_coefficients = [] #save the significant coefficients
significant_variable= [] #save the significant variables names (related to the significant coefficients)

significance=0.05 # alpha=0.05

# Generate histograms for each variable
for i in range(len(original_beta)):
    plt.hist(bootstrap_betas[:, i], bins=30,color='lightblue', edgecolor='black')
    plt.axvline(x=original_beta[i], color='red', linestyle='--', label=f'Original Beta: {original_beta[i]:.4f}')
    
    # Defining the confidence interval
    lower_bound = np.percentile(bootstrap_betas[:, i], (significance / 2) * 100) 
    upper_bound = np.percentile(bootstrap_betas[:, i], (1 - significance / 2) * 100) 

    plt.axvline(x=upper_bound, color='blue', linestyle='-', label=f'Upper Bound: {upper_bound:.4f}')
    plt.axvline(x=lower_bound, color='green', linestyle='-', label=f'Lower Bound: {lower_bound:.4f}')

    #Check if the variable is significant for the model
    if not (lower_bound <= 0 <= upper_bound):
        #save the significant coefficients and the significant variables
        significant_variable.append(variables[i])
        significant_coefficients.append(original_beta[i])

    # Plot the distribution
    plt.title(f'Histogram of Bootstrap Samples for {variables[i]}')
    plt.xlabel('Beta Coefficient')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()

# (v.)

# Print the significant coefficients and the significant variables on terminal
print("Significant coefficients:", significant_coefficients)
print("Significant variables:", significant_variable)
    
# Find significance variables indexes
significant_indices = [variables.index(variable) for variable in significant_variable]

#Save the significant variables dataset in an array both for training and for test sets
X_train_significant = X_train[:, significant_indices] # training set
X_test_significant = X_test[:, significant_indices] # test set

# Readapting the model to the significant variables
beta_significant = gradient_descent(X_train_significant, y_train, alpha, epsilon, max_iter)

# Print the significant coefficients (beta coefficients of the new model)
print("Beta coefficients (new model) :", beta_significant)

# Predict on the test set
y_pred_significant = predict(X_test_significant, beta_significant) >= 0.5


# Evaluating the model
true_positive = np.sum((y_test == 1) & (y_pred_significant == 1))
true_negative = np.sum((y_test == 0) & (y_pred_significant == 0))
false_positive= np.sum((y_test == 0) & (y_pred_significant == 1))
false_negative= np.sum((y_test == 1) & (y_pred_significant == 0))



# Print the performances
print("False Positive Rate (FPR):", false_positive / (false_positive + true_negative))
print("False Negative Rate (FNR):", false_negative/ (false_negative+ true_positive))

