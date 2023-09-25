# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load your dataset, assuming it's in a CSV file
data = pd.read_csv('ergoq_data_challenge.tsv', sep='\t')

def PCA1_PCA2(data):
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
