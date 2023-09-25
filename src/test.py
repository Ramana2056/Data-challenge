# Import necessary libraries
import pandas as pd
from Model import dimensionality_reduction
from visualization import visualization_histograms
from visualization import visualization_boxplots
from visualization import visualization_scatter_plot
from visualization import visualization_cv_test
from anamolies_detection import detect_anomalies
from PCA_plotting import PCA1_PCA2
from Model import Classifier_Model

# Load the data
df = pd.read_csv('ergoq_data_challenge.tsv', sep='\t')

"""
TASK-1
"""
# Data Exploration

# Summary statistics for variable columns
summary_stats = df.describe()


'''
We have gone for the top 20 features here for these two reasons:
We are keeping computation resources in check.
If we go for many features it would be hard for us to understand the visualization.
'''

selected_data = dimensionality_reduction(df,20)

# Summary statistics for top 20 columns
(selected_data.describe())

#PLOTTING:

'''
histograms : 
Here we are plotting the histograms for out float valued columns. 
We used different colors or styles to distinguish between the "class" values 0 and 1. 
This will help us to see if there are differences in the distributions between the two classes.
'''
featured_data = selected_data.drop(columns=['class'])

visualization_histograms(selected_data,featured_data)

''''
Boxplots:
Box Plots: We create box plots to visualize the distribution of our float-valued columns by the "class" variable. 
This will help us to identify outliers and differences in the spread and central tendency between the two classes.
'''

visualization_boxplots(selected_data,featured_data)


'''
Scatter Plot Matrix: Here we are trying to know the relationship between the pairs of columns. 
This can help you identify any patterns or anomalies in the data.
'''

visualization_scatter_plot(selected_data)


# Detect anomalies with the default contamination parameter (5%)
anomalies_mask = detect_anomalies(featured_data)

# Filter the dataset to keep only the non-anomalous rows
clean_data = featured_data[~anomalies_mask]

# You can also choose to keep the anomalous rows separately if needed
anomalies = featured_data[anomalies_mask]

# Now, 'clean_data' contains the rows without anomalies, and 'anomalies' contains the anomalous rows.

"""
TASK-2
"""
PCA1_PCA2(df)

"""
TASK-3

We have selected random classifier for below reasons:
Rationale:
1. Robustness to Feature Space: Random Forests are generally robust to outliers and can handle features with varying scales,
 making them a good choice for datasets where feature engineering is not trivial.
2. Ensemble Learning: Random Forests are an ensemble learning method that combines multiple decision trees to improve 
prediction accuracy and reduce overfitting. This is particularly beneficial when dealing with complex datasets.
3. Feature Importance: Random Forests provide a built-in feature importance measure. Given that you've already selected a 
subset of variables, this can help verify whether your chosen features are indeed relevant for classification.
4. Handling Imbalanced Data: Random Forests can handle imbalanced datasets better than some other algorithms. 
The classifier can still perform well on minority class samples.
"""

''''
Here we are trying to see what dimensions fits the data for that case we have taken values in [5,10,15,20,25,30,35,40,45,50]
We will plot the test and validation accuracy and see which one to take
'''
dim = [5,10,15,20,25,30,35,40,45,50]
cv_scores,test_scores,n, rf_classifier = Classifier_Model(df,dim)
visualization_cv_test(n,cv_scores,test_scores)

#By the above ploting, we can go with 15 features as the difference between test and validation accuracy is very less.
# As the dimensions are lower, we can train the the data very quickly even if we have larger dataset.

cv_scores,test_scores,n, final_rf_classifier = Classifier_Model(df,15)

final_rf_classifier