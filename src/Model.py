import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier


def dimensionality_reduction(df,n):
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

  # Concatenate the selected features with the target column
  selected_data = pd.concat([selected_features, y], axis=1)
  return selected_data



# Load your dataset
df = pd.read_csv('ergoq_data_challenge.tsv', sep='\t')
n = [5,10,15,20,25,30,35,40,45,50]
cv_scores = []
test_scores = []

def Classifier_Model(df,n):
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
      # print("Cross-Validation Accuracy for", i ," : {:.2f}%".format(np.mean(cross_val_scores) * 100))
      # print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
      cv_scores.append(np.mean(cross_val_scores) * 100)
      test_scores.append(test_accuracy * 100)
    return cv_scores,test_scores,n, rf_classifier