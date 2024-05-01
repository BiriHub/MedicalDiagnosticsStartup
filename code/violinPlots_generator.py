###
# Code for the diagnostic tool 2° practice part of the assignment, exercise 7.b
# Author: Birindelli Leonardo 
###

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import seaborn as sns

# (i.)

# Load the dataset
data = pd.read_csv('diabetes.csv') 

# List of variables to plot
variables = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']

# Setting up the property for plots
sns.set(style="whitegrid")

# Loop to create and display violin plots for each variable
for var in variables:
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Outcome', y=var, data=data,palette="Set2")
    plt.title(f'Violin Plot of {var} by Diabetes Outcome')
    plt.show()

#(iii.)

# Creating a correlation matrix
correlation_matrix = data.corr()

# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Variables")
plt.show()