import matplotlib.pyplot as plt
import seaborn as sns

def visualization_histograms(selected_data,featured_data):
    for column in featured_data.columns:
        plt.figure(figsize=(8, 6))
        plt.hist(selected_data.loc[selected_data['class'] == 0, column], bins=30, alpha=0.5, label='class 0')
        plt.hist(selected_data.loc[selected_data['class'] == 1, column], bins=30, alpha=0.5, label='class 1')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

def visualization_boxplots(selected_data,featured_data):
    for column in featured_data.columns:
        plt.figure(figsize=(8, 6))
        selected_data.boxplot(column=column, by='class')
        plt.title(f'Boxplot of {column} by class')
        plt.ylabel(column)
        plt.xlabel('class')
        plt.show()

def visualization_scatter_plot(selected_data):
    sns.pairplot(selected_data, hue='class', diag_kind='kde')
    plt.show()

def visualization_cv_test(n,cv_scores,test_scores):
    # Sample data
    n_values = n  # Replace with your 'n' values
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
