# -*- coding: utf-8 -*-

from google.colab import files
data_upload = files.upload()

"""reading the data of the files, with 'read_csv' function, and import the libraries will use."""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score


# upload the 'communities.data' into a data frame

data_frame = pd.read_csv('communities.data')


# reading the names file, to read the line that start with '@attribute'.
names_file = open('communities.names', 'r')

"""read the lines that start with '@attribute' into an array, and set them as column names."""

columns_name = []

for line in names_file:
    if line.startswith('@attribute'):

        #add the word into the array.
        columns_name.append(line.split()[1])



# set the names of data frame columns.
data_frame.columns = columns_name

"""function's that will use in the code."""

def clean_data_frame(data_frame):
      """
      at this function we will clean the data frame from missing values, and check if all the values is between 0-1.
      """

      cleand_columns = data_frame.apply(lambda col: col.notna().all() and col.between(0, 1).all(), axis=0)
      data_frame = data_frame.loc[:, cleand_columns]

      return data_frame


def print_indicators(y_test, y_pred,algho_name):
    """
    this function will calcuate the Indicators of the data and print thim.
    """

    print(f"\n alghoritm name : {algho_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall is: {recall_score(y_test, y_pred):.2f}")
    print(f"F1 Grade is: {f1_score(y_test, y_pred):.2f}")

"""delete the first 5 columns and also delete the missing values."""

# delete the first 5 columns with 'iloc'.
data_frame = data_frame.iloc[:, 5:]



# delete the missing values with 'loc' function.
clear_columns = data_frame.apply(lambda col: '?' not in col.values, axis=0)
data_frame = data_frame.loc[:, clear_columns]


# clean the data frame and check that all the values is between 0-1.
data_frame = clean_data_frame(data_frame)



# print the number columns and rows of the data frame.
print(f'number of data frame columns: {data_frame.shape[1]}')
print(f'number of data frame rows: {data_frame.shape[0]}')

"""calcuate the correlations of the 'ViolentCrimesPerPop' column"""

target_column = 'ViolentCrimesPerPop'
correlations = {}


for col in data_frame.columns:

    if col != target_column:
        data_to_calc_correlations = data_frame[[col, target_column]].dropna()


        # calcuate the correlations with 'pearsonr' function, and check first if the data is not empty.
        if not data_to_calc_correlations.empty:
            corr, _ = pearsonr(data_to_calc_correlations[col], data_to_calc_correlations[target_column])
            correlations[col] = corr


# sort the values and create a pnadas.Series
correlation_series = pd.Series(correlations).sort_values(key=abs ,ascending=False)


# calcuate how many variables have 'R' > 0.5.
num_of_bigger_corr = sum(abs(correlation_series) > 0.5)


print(f'number of variables with |r| > 0.5: {num_of_bigger_corr}')

"""create a plot that show the absolute values of the correlation in order"""

plt.figure(figsize=(14, 8))
plt.plot(correlation_series.index, correlation_series.abs().values, marker='o', linestyle='-', color='b', alpha = 0.7)


# set the title a Xlabel and Ylabel of the plot.
plt.title('Absolute Correlations with ViolentCrimesPerPop', fontsize=15)
plt.xlabel('Variables (Ordered by the Correlation)', fontsize=11)
plt.ylabel('Correlation', fontsize=11)

#remove the X-axis labels.
plt.xticks(ticks=[], labels=[], fontsize=8)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

"""Create a figure consisting of 12 scatter plots, where the Y-axis is ViolentCrimesPerPop and the X-axes are the first 12 variables."""

x_axis_var = data_frame.columns[:12]
y_axis_var = 'ViolentCrimesPerPop'


# create the plots
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 12))
axes = axes.flatten()


for i, var in enumerate(x_axis_var):
    axes[i].scatter(data_frame[var], data_frame[y_axis_var])
    axes[i].set_xlabel(var)
    axes[i].set_ylabel(y_axis_var)
    axes[i].set_title(f'Scatter Plot: {var} vs {y_axis_var}', fontsize=10)
    axes[i].grid(alpha=0.3)


# show the plot's
plt.tight_layout()
plt.show()

"""create a linear moudle 'MLR' that use all the varibales to Prediction the 'ViolentCrimesPerPop' column."""

numirc_data_frame = data_frame.select_dtypes(include=['number'])

# Separating the X and Y variables
x_var = numirc_data_frame.drop(columns=['ViolentCrimesPerPop'], errors='ignore')
y_var = numirc_data_frame['ViolentCrimesPerPop']


# add a constant column and Fitting a linear regression model.
X = sm.add_constant(x_var)
model = sm.OLS(y_var, X).fit()


print(model.summary())
adjusted_R_squared = model.rsquared_adj
print(f'adjusted R-squared: {adjusted_R_squared}')

"""Creating a model with the 12 variables with the highest correlation"""

# calcuate the correlation for the varibles.
correlation_arr = data_frame.corr()
correlation_with_ViolentCrimesPerPop = correlation_arr['ViolentCrimesPerPop']


# sort the array to get the highest 12 varibles, and choose them.
correlation_of_12_variables = correlation_with_ViolentCrimesPerPop.drop('ViolentCrimesPerPop').abs().sort_values(ascending=False).head(12).index # Get the index (column names)
x_12_varibales = data_frame[correlation_of_12_variables] # Use the column names for selection


y_12_varibales = data_frame['ViolentCrimesPerPop']


# adding a constant and create a linear moudle.
x_12_adding_constant = sm.add_constant(x_12_varibales)
model_12_varibales = sm.OLS(y_12_varibales, x_12_adding_constant).fit()

print(model_12_varibales.summary())
adjusted_R_squared = model_12_varibales.rsquared_adj

"""create a new column that show the status of the 'ViolentCrimesPerPop' column from the median."""

# calcuate the median of 'ViolentCrimesPerPop' column.
crime_median = data_frame['ViolentCrimesPerPop'].median()


# create a new column that show the status of the 'ViolentCrimesPer
data_frame['high crime'] = data_frame['ViolentCrimesPerPop'].apply(lambda x: 1 if x > crime_median else 0)

""" delete the 'ViolentCrimesPerPop' columm"""

data_frame = data_frame.drop(columns=['ViolentCrimesPerPop'], errors='ignore')

"""calcuate the 'Accuracy', 'Precision', 'Recall' and 'F1 Grade'."""

x = data_frame.drop(columns=['high crime'], errors='ignore')
y = data_frame['high crime']



x_training_set, x_test_set, y_training_set, y_test_set = train_test_split(x, y, test_size=0.3, random_state=42)



# first alghoritm : Random Forrest
random_forest_alghoritm = RandomForestClassifier(random_state=42)
random_forest_alghoritm.fit(x_training_set, y_training_set)
y_pred_random_forest = random_forest_alghoritm.predict(x_test_set)



# second alghortim : Decision Tree
decision_tree_alghoritm = DecisionTreeClassifier(random_state=42)
decision_tree_alghoritm.fit(x_training_set, y_training_set)
y_pred_decision_tree = decision_tree_alghoritm.predict(x_test_set)



# calcuate the Indicators of the data by 'print_indicators' function.
print_indicators(y_test_set, y_pred_random_forest, 'Random Forrest')
print_indicators(y_test_set, y_pred_decision_tree, 'Decision Tree')

"""create a confusion matrix for each alghoritm"""

# create the confusion matrix for 'random_forest' and 'decision_tree' alghorim's.
random_forest_confusion_matrix = confusion_matrix(y_test_set, y_pred_random_forest)
decision_tree_confusion_matrix = confusion_matrix(y_test_set, y_pred_decision_tree)


# show the plot of the confusion_matrix for each alghoritm, and set a title for each plot, and show thim.
ConfusionMatrixDisplay(confusion_matrix=random_forest_confusion_matrix, display_labels=["Low Crime", "High Crime"]).plot()
plt.title("Confusion Matrix: Random Forest")


# show the plot of the confusion_matrix for each alghoritm, and set a title for each plot, and show thim.
ConfusionMatrixDisplay(confusion_matrix=decision_tree_confusion_matrix, display_labels=["Low Crime", "High Crime"]).plot()
plt.title("Confusion Matrix: Decision Tree")


# show the matrix.
plt.show()

"""Creating an ROC Curve for each alghoritm and calcuate the AUC value."""

# calcuate the probabilities for the 'random forset' and 'decision tree' alghoritm.
random_forest_alghoritm_probabilities = random_forest_alghoritm.predict_proba(x_test_set)[:, 1]
decision_tree_alghoritm_probabilities = decision_tree_alghoritm.predict_proba(x_test_set)[:, 1]


random_forest_alghoritm_fpr, random_forest_alghoritm_tpr, _ = roc_curve(y_test_set, random_forest_alghoritm_probabilities)
decision_tree_alghoritm_fpr, decision_tree_alghoritm_tpr, _ = roc_curve(y_test_set, decision_tree_alghoritm_probabilities)


# calcuate the 'AUC' value for each alghoritm.
random_forest_alghoritm_auc = roc_auc_score(y_test_set, random_forest_alghoritm_probabilities)
decision_tree_alghoritm_auc = roc_auc_score(y_test_set, decision_tree_alghoritm_probabilities)


# create a ROC Curve plot.
plt.figure(figsize=(11, 7))
plt.plot(random_forest_alghoritm_fpr, random_forest_alghoritm_tpr, label=f'Random Forest (AUC = {random_forest_alghoritm_auc:.2f}')
plt.plot(decision_tree_alghoritm_fpr, decision_tree_alghoritm_tpr, label=f'Decision Tree (AUC = {decision_tree_alghoritm_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing (AUC = 0.5)", linewidth=1)


# set a title a X-axis label and Y-axis Label.
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison: Random Forest vs Decision Tree')

#set a grid for the plot and show the plot.
plt.grid(alpha=0.3)
plt.legend()
plt.show()
