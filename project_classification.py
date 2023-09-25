# -*- coding: utf-8 -*-
"""Welcome To Colaboratory

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/notebooks/intro.ipynb

#Task-1
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv('ergoq_data_challenge.tsv', sep='\t')

data

# Data Exploration

# Summary statistics for variable columns
summary_stats = data.describe()

# Load the data
df = pd.read_csv('ergoq_data_challenge.tsv', sep='\t')

"""###We have gone for the top 20 features here for these two reasons:
####We are keeping computation resources in check.
####If we go for many features it would be hard for us to understand the visualization.


"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Assuming you have your data in a DataFrame df and the target column in target_col
# Separate the features and target
X = df.drop(columns=['class'])
y = df['class']

# Handle missing values (impute)
imputer = SimpleImputer(strategy='mean')  # You can choose other strategies as well
X_imputed = imputer.fit_transform(X)

# Standardize the features (PCA is sensitive to scale)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Perform PCA to reduce dimensionality
n_components = 20  # Number of top components you want to keep
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

# You can now use X_pca as your reduced feature matrix

# To find the explained variance ratio for each component
explained_variance_ratio = pca.explained_variance_ratio_

# If you want to see the importance of original features in the first component
component_importance = np.abs(pca.components_[0])

# To get the column names of the top 20 features
top_20_feature_indices = np.argsort(component_importance)[-20:]
top_20_features = X.columns[top_20_feature_indices]

# You can now select these top 20 features from your original dataset
selected_features = X[top_20_features]

# Optionally, you can also concatenate the selected features with the target column
selected_data = pd.concat([selected_features, y], axis=1)

print(top_20_features)

selected_data.describe()

"""##Visualization

Here we are plotting the histograms for out float valued columns. We used different colors or styles to distinguish between the "class" values 0 and 1. This will help us to see if there are differences in the distributions between the two classes.
"""

featured_data = selected_data.drop(columns=['class'])

import matplotlib.pyplot as plt

for column in featured_data.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(selected_data.loc[selected_data['class'] == 0, column], bins=30, alpha=0.5, label='class 0')
    plt.hist(selected_data.loc[selected_data['class'] == 1, column], bins=30, alpha=0.5, label='class 1')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

"""*Box Plots*: We create box plots to visualize the distribution of our float-valued columns by the "class" variable. This will help us to identify outliers and differences in the spread and central tendency between the two classes."""

for column in featured_data.columns:
    plt.figure(figsize=(8, 6))
    selected_data.boxplot(column=column, by='class')
    plt.title(f'Boxplot of {column} by class')
    plt.ylabel(column)
    plt.xlabel('class')
    plt.show()

"""Scatter Plot Matrix: Here we are trying to know the relationship between the pairs of columns. This can help you identify any patterns or anomalies in the data."""

import seaborn as sns

sns.pairplot(selected_data, hue='class', diag_kind='kde')
plt.show()

from sklearn.ensemble import IsolationForest
import numpy as np

def detect_anomalies(data, contamination=0.05, random_state=None):
    """
    Detect anomalies in a dataset using Isolation Forest.

    Parameters:
        data (numpy.ndarray or pandas.DataFrame): Input data with shape (n_samples, n_features).
        contamination (float, optional): The proportion of outliers in the dataset. Default is 0.05 (5%).
        random_state (int or RandomState, optional): Seed for the random number generator. Default is None.

    Returns:
        numpy.ndarray: An array of boolean values indicating whether each sample is an anomaly (True) or not (False).
    """
    # Create the Isolation Forest model
    model = IsolationForest(contamination=contamination, random_state=random_state)

    # Fit the model to the data and make predictions
    if isinstance(data, np.ndarray):
        anomalies = model.fit_predict(data)
    elif isinstance(data, pd.DataFrame):
        anomalies = model.fit_predict(data.values)
    else:
        raise ValueError("Input data must be a numpy array or pandas DataFrame.")

    # Convert predictions to boolean values (True for anomalies, False for normal)
    return anomalies == -1

# Load your dataset into a pandas DataFrame, assuming it's named 'data'
# Ensure 'data' contains your 20 columns of float values in the range (-2, 2)

# Detect anomalies with the default contamination parameter (5%)
anomalies_mask = detect_anomalies(featured_data)

# Filter the dataset to keep only the non-anomalous rows
clean_data = featured_data[~anomalies_mask]

# You can also choose to keep the anomalous rows separately if needed
anomalies = featured_data[anomalies_mask]

# Now, 'clean_data' contains the rows without anomalies, and 'anomalies' contains the anomalous rows.

anomalies

"""#Task-2"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Load your dataset, assuming it's in a CSV file
data = pd.read_csv('ergoq_data_challenge.tsv', sep='\t')

# Extract the features (columns other than the target) and the target column

X = data.drop(columns=['class'])
y = data['class']

# Handle missing values (impute)
imputer = SimpleImputer(strategy='mean')  # You can choose other strategies as well
X_imputed = imputer.fit_transform(X)

# Standardize the features (PCA is sensitive to scale)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)


# Perform PCA to reduce dimensionality to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame with the PCA results
pca_df = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
pca_df["class"] = y  # Add the target column back to the DataFrame

# Plot the PCA results
plt.figure(figsize=(10, 6))
colors = {0: 'blue', 1: 'red'}
markers = {0: 'o', 1: 's'}

for class_val in [0, 1]:
    subset = pca_df[pca_df["class"] == class_val]
    plt.scatter(subset["PC1"], subset["PC2"], c=colors[class_val], label=f"Class {class_val}", marker=markers[class_val])

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.title("PCA Plot of Samples (PC1 vs. PC2)")
plt.grid(True)
plt.show()

"""#Task -3

##We have selected random classifier for below reasons:

####Rationale:
####Robustness to Feature Space: Random Forests are generally robust to outliers and can handle features with varying scales, making them a good choice for datasets where feature engineering is not trivial.

####Ensemble Learning: Random Forests are an ensemble learning method that combines multiple decision trees to improve prediction accuracy and reduce overfitting. This is particularly beneficial when dealing with complex datasets.

####Feature Importance: Random Forests provide a built-in feature importance measure. Given that you've already selected a subset of variables, this can help verify whether your chosen features are indeed relevant for classification.

####Handling Imbalanced Data: Random Forests can handle imbalanced datasets better than some other algorithms. The classifier can still perform well on minority class samples.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def dimensionality_reduction(df,n):
  # Assuming you have your data in a DataFrame df and the target column in target_col
  # Separate the features and target
  X = df.drop(columns=['class'])
  y = df['class']

  # Handle missing values (impute)
  imputer = SimpleImputer(strategy='mean')  # You can choose other strategies as well
  X_imputed = imputer.fit_transform(X)

  # Standardize the features (PCA is sensitive to scale)
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X_imputed)

  # Perform PCA to reduce dimensionality
  n_components = n  # Number of top components you want to keep

  pca = PCA(n_components=n_components)
  X_pca = pca.fit_transform(X_scaled)

  # You can now use X_pca as your reduced feature matrix

  # To find the explained variance ratio for each component
  explained_variance_ratio = pca.explained_variance_ratio_

  # If you want to see the importance of original features in the first component
  component_importance = np.abs(pca.components_[0])

  # To get the column names of the top n features
  top_n_feature_indices = np.argsort(component_importance)[-n:]
  top_n_features = X.columns[top_n_feature_indices]

  # You can now select these top n features from your original dataset
  selected_features = X[top_n_features]

  # Optionally, you can also concatenate the selected features with the target column
  selected_data = pd.concat([selected_features, y], axis=1)
  return selected_data

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Load your dataset
df = pd.read_csv('ergoq_data_challenge.tsv', sep='\t')
n = [5,10,15,20,25,30,35,40,45,50]
cv_scores = []
test_scores = []

for i in n:
  data = dimensionality_reduction(df,i)

  # Separate features (X) and target (y)
  X = data.drop("class", axis=1)
  y = data["class"]


  # Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Initialize the Random Forest Classifier
  rf_classifier = RandomForestClassifier(random_state=42)

  # Evaluate the classifier using cross-validation
  cross_val_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5, scoring='accuracy')

  # Fit the classifier on the training data
  rf_classifier.fit(X_train, y_train)

  # Assess the classifier's performance on the test set
  test_accuracy = rf_classifier.score(X_test, y_test)

  # Print the cross-validation and test accuracy
  print("Cross-Validation Accuracy for", i ," : {:.2f}%".format(np.mean(cross_val_scores) * 100))
  print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
  cv_scores.append(np.mean(cross_val_scores) * 100)
  test_scores.append(test_accuracy * 100)

import matplotlib.pyplot as plt
import numpy as np

# Sample data
n_values = [5,10,15,20,25,30,35,40,45,50]  # Replace with your 'n' values
validation_accuracy = cv_scores  # Replace with your validation accuracy values
test_accuracy = test_scores  # Replace with your test accuracy values

# Create a line plot
plt.figure(figsize=(8, 6))
plt.plot(n_values, validation_accuracy, marker='o', label='Validation Accuracy', linestyle='-', color='b')
plt.plot(n_values, test_accuracy, marker='o', label='Test Accuracy', linestyle='-', color='g')

# Add labels and a legend
plt.xlabel('Value of n')
plt.ylabel('Accuracy')
plt.title('Validation and Test Accuracy vs. Value of n')
plt.legend()

# Add a grid for better visualization (optional)
plt.grid(True)

# Show the plot
plt.show()

"""By the above ploting, we can go with 15 features as the difference between test and validation accuracy is very less. As the dimensions are lower, we can train the the data very quickly even if we have larger dataset."""





