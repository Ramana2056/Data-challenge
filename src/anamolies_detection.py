from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd

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