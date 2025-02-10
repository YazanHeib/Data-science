# -*- coding: utf-8 -*-

from google.colab import files
files_uploaded = files.upload()

"""import the libraries will use in the code."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import math
import warnings
import numpy as np
from scipy.stats import kruskal, hypergeom
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

"""upload the 'lewis.csv' file into a data frame, and clean the data according to the instructions."""

# read the file, and set the 'Sample' column to the index of the data frame.
lewis_data_frame = pd.read_csv('lewis.csv', index_col='Sample')



# get all the columns that the childreen's get Antibiotics or Steroids.
data_will_remove = lewis_data_frame.loc[(lewis_data_frame['Antibiotics'] != 'Not.Use') | (lewis_data_frame['Steroids'] != 'Not.Use'), 'Subject'].unique()



# clean the 'lewis_data_drame' from removed data.
lewis_data_frame = lewis_data_frame[~lewis_data_frame['Subject'].isin(data_will_remove)]



# print the number of Samples that Remaining for each combination
samples_data_count = lewis_data_frame.groupby(['Time', 'Treatment']).size().reset_index(name='Sample Count')


# show the result of the samples.
print(samples_data_count)

"""part B.1 : check the effect of treatments on the disease."""

# Question B.1

# create a data frame that will be contain the childreen's with Crohn’s.
child_with_Crohn_data = lewis_data_frame[lewis_data_frame['Treatment'] != 'Healthy']


# create a plot.
plt.figure(figsize=(13, 8))


# create a boxplot that his x-axis is Treatment, and the y is Value of FCP.
sns.boxplot(x='Time', y='FCP', data=child_with_Crohn_data, hue='Treatment', palette="Set2")



# set a title, X-axis and y-axis label.
plt.title('FCP Distribution for Different Treatments Across Time\n')
plt.xlabel('Treatment')
plt.ylabel('FCP')




# show the plot, and set a legend.
plt.legend(title="Treatment")
plt.tight_layout()
plt.show()

"""part B.2 : check the effect of treatments on the disease."""

# Question B.2


# create a list to add the P-values to it.
p_val_list = []


# gets all the unique time points from the 'Time' column.
time_points = child_with_Crohn_data['Time'].unique()


for time in time_points:

  # create a new data frame for each time point.
  data_for_time = child_with_Crohn_data[child_with_Crohn_data['Time'] == time]


  # create a groups for each treatment and time point, and Performing the Kruskal-Wallis test for the groups .
  treatment_group = [data_for_time[data_for_time['Treatment']  == treatment]['FCP'] for treatment in data_for_time['Treatment'].unique()]
  statistic, p_value = kruskal(*treatment_group)


  # add the P-value to the list.
  p_val_list.append(p_value)


  # applies the FDR correction to the p-values.
  _, adjusted_p_values, _, _ = multipletests(p_val_list, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)


# print the output
for time, p_value in zip(time_points, p_val_list):
  print(f'Time: {time}, Kruskal-Wallis P-value is: {p_value}')

"""part 3.1 : check the effect of treatments on Microbiome"""

# create a boxPlots that show the distribution of values of 'Shannon' for the childreens.


# filter the data to include the healthy children and also the childreen's with 'Crohn'.
childreen_data_for_boxplot = lewis_data_frame[['Time', 'Treatment', 'Shannon']]


# create a plot.
plt.figure(figsize=(13, 8))


sns.boxplot(x='Time', y='Shannon', hue='Treatment', data=childreen_data_for_boxplot, palette="husl")


# set the tilte and X-axis, Y-axis Label's.
plt.title('Shannon Distribution for Different Treatments Across Time\n')
plt.xlabel('Treatment')
plt.ylabel('Shannon')



# show the plot, and adding a legend.
plt.legend(title="Treatment")
plt.tight_layout()
plt.show()

def generate_data_p_value_table(data_frame,metric_name, column_name):

    """
    This function generates a table with treatment type, time, average metric value, P-value, Kruskal-Wallis P-value, and FDR-corrected P-value for each treatment group compared to healthy children.
    It takes a DataFrame, metric name, and column name as inputs.
    """

    # create a list for the data will calcuate.
    data_for_table = []

    # for each time point.
    for time in data_frame['Time'].unique():

        # geting the data for each time point.
        time_point_data = data_frame[data_frame['Time'] == time]

        # get the data of the healthy Childreen's.
        healthy_data = time_point_data[time_point_data['Treatment'] == 'Healthy']

        # get the data for the other treament's.
        for treatment in time_point_data['Treatment'].unique():
            if treatment != 'Healthy':
                treatment_data = time_point_data[time_point_data['Treatment'] == treatment][column_name]

                # performing the 'Kruskal-Wallis' test on the healthy group to the other groups.
                statistic, p_value = kruskal(healthy_data[column_name], treatment_data)

                # add the data to the list.
                data_for_table.append(
                    {'Treatment': treatment, 'Time': time, metric_name: healthy_data[column_name].mean(),
                     'P-value': p_value})

    # create a data frame for the data we had get.
    data_for_table = pd.DataFrame(data_for_table)

    # calcuate the 'P-value' for each Time point with kruskal, and add it to the data frame.
    statistic, p_value = kruskal(
        *[data_frame[data_frame['Treatment'] == treatment][column_name] for treatment in
          data_frame['Treatment'].unique()])
    data_for_table['P-value Kruskal-Walli'] = p_value

    # applies the FDR correction to the p-values.
    _, data_for_table['FDR corrected P-value'], _, _ = multipletests(data_for_table['P-value'], alpha=0.05,
                                                                     method='fdr_bh')

    # return data as a table.
    return data_for_table

"""part 3.3 : check the effect of treatments on Microbiome"""

# create anthoer plot and table for the Number of species.


# create a BoxPlot for Number of species.
plt.figure(figsize=(17, 11))


# Include 'NSpecies' in the DataFrame for the boxplot
childreen_data_for_boxplot = lewis_data_frame[['Time', 'Treatment', 'Shannon', 'NSpecies']]


# create a boxplot for each
sns.boxplot(x="Time", y="NSpecies", hue="Treatment", data=childreen_data_for_boxplot, palette="husl")


# set a tilte, Y-axis and X-axis Labels.
plt.title('Number of Species Distribution for Different Treatments Across Time\n')
plt.xlabel('Treatment')
plt.ylabel('Number of Species')



# show the plot, and adding a legend to the plot.
plt.legend(title="Treatment")
plt.tight_layout()
plt.show()

"""part 3.2 : check the effect of treatments on *Microbiome*"""

# Question 3.2
data_with_Mean_Shannon = generate_data_p_value_table(childreen_data_for_boxplot, 'Mean Shannon', 'Shannon')
data_with_Mean_Shannon

"""part 3.3 : check the effect of treatments on *Microbiome*"""

# Question 3.3
data_with_Mean_NSpecies_Treatment = generate_data_p_value_table(childreen_data_for_boxplot,  'Mean NSpecies Treatment', 'NSpecies')
data_with_Mean_NSpecies_Treatment

"""part 4.1: Examination of the microbiome without treatment."""

# reading the 'lewis.species.csv' file to data frame, and set the first column to the index of the data frame.
lewis_species_data_frame = pd.read_csv('lewis.species.csv', index_col=0)



# clear the data from all the row's that is not '1' or 'Healthy'.
cleaned_lewis_data_frame = lewis_data_frame[(lewis_data_frame['Time'] == '1') | (lewis_data_frame['Treatment'] == 'Healthy')]



# reset all the cells at 'lewis.species.csv' file that his is data is less than '0.1'.
cleaned_childreen_data = lewis_species_data_frame.mask(lewis_species_data_frame < 0.1, 0)


# at this part of code will clear all the columns that there cells, in less than 10% of the cells

# calcuate the limit 10% percent from the rows.
limit_of_ten_percent = 0.1 * cleaned_childreen_data.shape[0]


# calc all the columns that has a 90% of '0' value.
columns_to_remove = cleaned_childreen_data.columns[((cleaned_childreen_data != 0).sum())  < limit_of_ten_percent]


# remove the columns from the data frame.
cleaned_childreen_data.drop(columns=columns_to_remove, inplace=True)


# merge of the file 'lewis.species.csv' and lewis_data_frame to one data frame.
lewis_merged_data_frame = cleaned_lewis_data_frame.merge(cleaned_childreen_data, left_index=True, right_index=True, how='inner')

"""part 4.2 Examination of the microbiome without treatment."""

def microbe_gender_with_childreen(data_frame, M, n):
  """
  calcaute the P-value for microbe gender with the hypergeometric distribution,
  and applies the FDR correction to the p-values.
  """


  # allocate an array for the result.
  childreen_result_data = []

  # allocate an array for the p-value.
  p_values = []


  # choose the columns of the microb.
  microbe_columns = data_frame.columns[9:]

  for microbe in microbe_columns:


      # N: is the number of the sick children which the microbe is present.
      N = (data_frame[microbe] > 0 ).sum()

      # K: is the number of the helthy childreen's.
      K = ((data_frame['Treatment'] == 'Healthy') & (data_frame[microbe] > 0)).sum()


      # calcuate the p-Value with hypergeometric distribution if the N and K is bigger than 0.
      p_value = hypergeom.sf(K-1, M, N, n)# if (N > 0 and K > 0) else 1


      # add the result to the list.
      childreen_result_data.append({'Microbe': microbe, 'P-value': p_value})
      p_values.append(p_value)



  # create a data frame from the result.
  childreen_result_data_frame = pd.DataFrame(childreen_result_data)


  if not childreen_result_data_frame.empty:
      _, pvals_corrected, _, _ = multipletests(childreen_result_data_frame['P-value'], method='fdr_bh')
      childreen_result_data_frame['FDR_corrected_P-value'] = pvals_corrected

  return childreen_result_data_frame

"""part 4: Examination of the microbiome without treatment."""

childreen_result_data_frame = microbe_gender_with_childreen(lewis_merged_data_frame,63,26)
childreen_result_data_frame

"""part 5 : Clustering Samples Based on Microbial Composition"""

# all the columns that discreeb the microbe.
X = lewis_merged_data_frame.iloc[:, 9:]
Y = lewis_merged_data_frame['Treatment']


# split the table we had create in part D, into 70% training samples and 30% test samples, with random_state = 1984.
x_test, x_train, y_test, y_train = train_test_split(X, Y, test_size=0.3, random_state=1984)

"""part 5.2 : Clustering Samples Based on Microbial Composition"""

# create a two KNN models the first one with 'Euclidean distance' and the secind one with 'Jaccard distance', made predictions, and compared accuracy and confusion matrices.


# first model with the Euclidean distance.
knn_euclidean = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_euclidean.fit(x_train, y_train)


# predicting the test group.
y_pred_euclidean = knn_euclidean.predict(x_test)


# calcuate the 'Confusion' and 'Accuracy' matrix.
confusion_matrix_euclidean = confusion_matrix(y_test, y_pred_euclidean)
accuracy_euclidean = accuracy_score(y_test, y_pred_euclidean)


# prepare the data for 'Jaccard distance', and convert the values from 0 to 1.
X_train_jaccard = (x_train > 0).astype(bool).astype(int)
X_test_jaccard = (x_test > 0).astype(bool).astype(int)


# second model with the Jaccard distance.
knn_model_jaccard = KNeighborsClassifier(n_neighbors=5, metric='jaccard')
knn_model_jaccard.fit(X_train_jaccard, y_train)


# predicting the test group.
y_pred_jaccard = knn_model_jaccard.predict(X_test_jaccard)


# calcuate the 'Confusion' and 'Accuracy' matrix.
confusion_matrix_jaccard = confusion_matrix(y_test, y_pred_jaccard)
accuracy_jaccard = accuracy_score(y_test, y_pred_jaccard)


# show the data.
print("Confusion Matrix with Euclidean Distance:")
print(confusion_matrix_euclidean)
print("Accuracy with Euclidean Distance:", accuracy_euclidean)

"""part 5.3 : Clustering Samples Based on Microbial Composition"""

# create a new data frame that will get all the microbes in part D, that there 'FDR' value is less or equal than 0.1
microbe_data_frame = childreen_result_data_frame[childreen_result_data_frame['FDR_corrected_P-value'] > 0.1]['Microbe']


# save the spicfic columns for the training samples and test samples.
x_test_microbe = x_test[microbe_data_frame]
x_train_microbe = x_train[microbe_data_frame]

"""part 5.4 : Clustering Samples Based on Microbial Composition"""

# with the filterd data we had get, will do a new Comparison with the two knn modles.


# first model with the Euclidean distance.
knn_euclidean = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_euclidean.fit(x_train_microbe, y_train)


# predicting the test group.
y_pred_euclidean = knn_euclidean.predict(x_test_microbe)


# calcuate the 'Confusion' and 'Accuracy' matrix.
confusion_matrix_euclidean = confusion_matrix(y_test, y_pred_euclidean)
accuracy_euclidean = accuracy_score(y_test, y_pred_euclidean)



# prepare the data for 'Jaccard distance', and convert the values from 0 to 1.
X_train_jaccard = (x_train_microbe > 0).astype(bool).astype(int)
X_test_jaccard = (x_test_microbe > 0).astype(bool).astype(int)


# second model with the Jaccard distance.
knn_model_jaccard = KNeighborsClassifier(n_neighbors=5, metric='jaccard')
knn_model_jaccard.fit(X_train_jaccard, y_train)


# predicting the test group.
y_pred_jaccard = knn_model_jaccard.predict(X_test_jaccard)


# calcuate the 'Confusion' and 'Accuracy' matrix.
confusion_matrix_jaccard = confusion_matrix(y_test, y_pred_jaccard)
accuracy_jaccard = accuracy_score(y_test, y_pred_jaccard)


# show the data.
print("Confusion Matrix with Euclidean Distance:")
print(confusion_matrix_euclidean)
print("Accuracy with Euclidean Distance:", accuracy_euclidean)
