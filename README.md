# Data_challenge
## This project contains the following data:
## anamolied_detection -- detect_anomalies (used to detect anamolied and return an array of boolean values indicating whether each sample is an anomaly (True) or not (False))
## Model: dimensionality_reduction (reduces the dimensions by the top 'n' features)
##     : Classifier_Model() (which gives the model which can predict the output class)
## visualization: We have three visualization functions(histograms, box-plot and scatter)
##              : Along with it we have one more plot which can plot the test results of CV and test dataset by their dimensions
## PCA_plotting  : Using PCA, we have a scatter plot between top two important features.



### We have involved all the comments in each section. Kindly run test.py to get the results of all the tasks.

### We are including the findings of all the tasks here seperately as well.

### TASK-1 :
#### We have gone for the top 20 features here for these two reasons:
#### 1.We are keeping computation resources in check.
#### 2.If we go for many features it would be hard for us to understand the visualization.

#### Choosing the visualization plots:
#### histograms : 
#### Here we are plotting the histograms for out float valued columns. 
#### We used different colors or styles to distinguish between the "class" values 0 and 1. 
#### This will help us to see if there are differences in the distributions between the two classes.

#### Boxplots:
#### Box Plots: We create box plots to visualize the distribution of our float-valued columns by the "class" variable. 
#### This will help us to identify outliers and differences in the spread and central tendency between the two classes.

#### Scatter Plot Matrix: Here we are trying to know the relationship between the pairs of columns. 
#### This can help you identify any patterns or anomalies in the data.


### Task-3:


#### We have selected random classifier for below reasons:
#### Rationale:
#### 1. Robustness to Feature Space: Random Forests are generally robust to outliers and can handle features with varying scales,
#### making them a good choice for datasets where feature engineering is not trivial.
#### 2. Ensemble Learning: Random Forests are an ensemble learning method that combines multiple decision trees to improve 
#### prediction accuracy and reduce overfitting. This is particularly beneficial when dealing with complex datasets.
#### 3. Feature Importance: Random Forests provide a built-in feature importance measure. Given that you've already selected a 
#### subset of variables, this can help verify whether your chosen features are indeed relevant for classification.
#### 4. Handling Imbalanced Data: Random Forests can handle imbalanced datasets better than some other algorithms. 
#### The classifier can still perform well on minority class samples.

#### Here we are trying to see what dimensions fits the data for that case we have taken values in [5,10,15,20,25,30,35,40,45,50]
#### We will plot the test and validation accuracy and see which one to take

#### By the ploting we got , we can go with 15 features as the difference between test and validation accuracy is very less.
#### As the dimensions are lower, we can train the the data very quickly even if we have larger dataset.

#### final_rf_classifier is the final model using the optimal parameters.
