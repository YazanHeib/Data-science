# -*- coding: utf-8 -*-

from google.colab import files
data_upload = files.upload()

"""reading the data of the files, with 'read_csv' function, and import the libraries will use."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import math
import warnings
import numpy as np
from scipy.stats import linregress


# reading the files into a data frames.
primary_energy_supply_file = pd.read_csv('primary_energy_supply.csv')
world_population_file = pd.read_csv('world_population.csv')
gdp_file = pd.read_csv('gdp.csv')

"""delete all the row's that his 'MEASURE' is not 'MLN_TOE' or USD_CAD'."""

# select only the rows where 'MEASURE' equal to 'MLN_TOE'.
primary_energy_supply_file = primary_energy_supply_file[primary_energy_supply_file['MEASURE'] == 'MLN_TOE']


# select only the rows where 'MEASURE' equal to 'USD_CAD'.
gdp_file = gdp_file[gdp_file['MEASURE'] == 'USD_CAP']

"""this function will get a two data frames and update the energy data, to energy per capity"""

def update_energy_per_capity(primary_energy_supply_data_frame, popluation_data_frame):

      # rename the columns of 'Country Code' in the gdp file, to 'LOCATION'.
      popluation_data_frame = popluation_data_frame.rename(columns={'Country Code': 'LOCATION'})


      # reshape the population data to have 'TIME' data as a column.
      # 'melt' function in pandas, which is a powerful tool for converting a DataFrame from wide format to long format.
      popluation_data = popluation_data_frame.melt( id_vars=['LOCATION'], var_name='TIME', value_name='Population')


      primary_energy_supply_data_frame['TIME'] = primary_energy_supply_data_frame['TIME'].astype(str)
      popluation_data['TIME'] = popluation_data['TIME'].astype(str)

      # merge the primary energy supply data frame, with the gdp reshaped data frame.
      merge_popluation_primary_data = primary_energy_supply_data_frame.merge(popluation_data, on=['LOCATION', 'TIME'], how='left')


      # merge the primary energy supply data frame, with the gdp reshaped data frame.
      merge_popluation_primary_data = primary_energy_supply_data_frame.merge( popluation_data, on=['LOCATION', 'TIME'], how='left')


      # calucate the energy supply per capity.
      merge_popluation_primary_data['Energy_Supply_per_Capita'] = merge_popluation_primary_data['Value'] / merge_popluation_primary_data['Population']


      return merge_popluation_primary_data

"""this function will Clean all the combinations that do not have data for them and also print them."""

def delete_years_without_data(primary_energy_supply_data_frame, popluation_data_frame):


    # reshape the popluation data to have 'TIME' data as a column.
    popluation_data = popluation_data_frame.melt( id_vars=['Country Code'], var_name='TIME', value_name='Population')


    # create a Set of the 'Country Code' and 'TIME' column's from popluation data frame.
    population_set_location_time = set(zip(popluation_data['Country Code'], popluation_data['TIME']))


    # create a Set of the 'LOCATION' and 'TIME' column's from primary energy supply data frame.
    primary_set_location_time = set(zip(primary_energy_supply_data_frame['LOCATION'], primary_energy_supply_data_frame['TIME']))


    primary_energy_supply_data_frame['TIME'] = primary_energy_supply_data_frame['TIME'].astype(str)
    popluation_data['TIME'] = popluation_data['TIME'].astype(str)


    # filtering the primary_energy_supply_data_frame, and keeping only rows where the 'LOCATION, TIME' pair is in popluation_data_frame.
    filtered_primary_energy_supply_file = primary_energy_supply_data_frame[primary_energy_supply_data_frame.apply(
    lambda row: (row['LOCATION'], row['TIME']) in population_set_location_time, axis=1)]


    # Finding the 'LOCATION, TIME' pairs in primary_energy_supply_data_frame that are not in popluation_data_frame
    no_data_Time_Location = primary_set_location_time - population_set_location_time


    # get the 'Countries' and 'Years' that we don't have anu data about.
    contries = {no_data_Time_Location_pair[0] for no_data_Time_Location_pair in no_data_Time_Location}
    years = {no_data_Time_Location_pair[1] for no_data_Time_Location_pair in no_data_Time_Location}


    # sort the countries and the years that don't have data
    sorted_countries = sorted(set(contries))


    print("Countries that don't have any data about : ",', '.join(sorted_countries))
    print("Years that don't have any data about : ", years,'\n')


    return filtered_primary_energy_supply_file

"""will delete the 'LOCATION' and 'TIME' data that not appers in the two data frames, and calcuate the energy supply per capity."""

# clear the 'primary_energy_supply_file' with function 'delete_years_without_data'.
clear_energy_supply = delete_years_without_data(primary_energy_supply_file, world_population_file)


# Ensure 'TIME' is string in clear_energy_supply
#clear_energy_supply['TIME'] = clear_energy_supply['TIME'].astype(str)
gdp_file['TIME'] = gdp_file['TIME'].astype(str)


# Remove 'TIME' and 'LOCATION' data that appear in the first table but not in the gdp table.
clear_energy_supply = clear_energy_supply.merge(gdp_file[['TIME', 'LOCATION']], on=['TIME', 'LOCATION'], how='inner')


# Calculate the energy supply per capita with the 'update_energy_per_capity' function.
energy_per_capity = update_energy_per_capity(clear_energy_supply, world_population_file)

"""create a data frame that containes the 'Location', 'Year', 'energy_per_capity' and gdp"""

#create a new data frame for the spicfic columns.
final_data_frame = pd.DataFrame()


# copy the data of 'LOCATION', 'TIME' and 'Energy_Supply_per_Capita'.
final_data_frame['LOCATION'] = energy_per_capity['LOCATION']
final_data_frame['TIME'] = energy_per_capity['TIME']
final_data_frame['Energy_Supply_per_Capita'] = energy_per_capity['Energy_Supply_per_Capita']


# copy the year that appers in the dataframe, and having data about.
final_data_frame = final_data_frame.merge(gdp_file[['LOCATION', 'TIME', 'Value']],on=['LOCATION', 'TIME'],how='left')

"""function's that will use in the code."""

def calc_array_square(arr):
    """
    this function will calculate the square of the array.
    """
    length = math.ceil(math.sqrt(len(arr)))
    return length


def edit_the_data(data_frame):
    """
    this fnuction will edit the 'TIME' column of a data frame, by converit it to numeric.
    """
    data_frame['TIME'] = pd.to_numeric(data_frame['TIME'], errors='coerce')
    return data_frame


def plot_scatter_create(ax, x, y, data_frame, plot_color_map=None, norm=None):
    """
       this function will create a scatter plot with the given axis.
    """
    scatter_plot = ax.scatter(data_frame[x], data_frame[y], c=data_frame['TIME'], cmap=plot_color_map, norm=norm)
    return scatter_plot


def clean_the_data(data_frame, columns):
    """
    clean the data by convert specified columns to numeric and drop rows with missing values in those columns.
    """
    for col in columns:
        data_frame[col] = pd.to_numeric(data_frame[col], errors='coerce')
    return data_frame.dropna(subset=columns)



def add_regression_line(ax, x, y, data_frame):
    """
    this function will add a regression line to the scatter plot.
    """
    sns.regplot(x=x, y=y, data=data_frame, scatter=False, ax=ax, line_kws={"color": "red", "linewidth": 1.5})


def set_the_plot_titles(ax, title, xlabael, ylabel):
    """
    this function will set a title and xlabel, ylabel for a given plot.
    """
    ax.set_title(title)
    ax.set_xlabel(xlabael, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)


def add_years_to_plot(ax, min_year, max_year):
    """
    this function will add text annotations for the first and last year of the given data.
    """
    ax.text(0.05, 0.90, f"First Year : {min_year}\nLast Year : {max_year}",
            transform=ax.transAxes, fontsize=8, bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))


def add_colormap_to_plot(fig, ax, color_map, norm):
    """
    this function will add a colorbar to the plot.
    """
    color_bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=color_map), ax=ax, orientation='vertical',
                             fraction=0.046, pad=0.04)
    color_bar.set_label('TIME', fontsize=8)
    color_bar.ax.tick_params(labelsize=7)



def calculate_correlations(data_frame):
    """
    this function will calculate the correlation between the 'Energy_Supply_per_Capita' and 'Value' columns of a data frame.
    """

    data_correlations = data_frame.groupby('LOCATION').apply(lambda X: X[['Energy_Supply_per_Capita', 'Value']].corr().iloc[0, 1])
    return data_correlations.dropna()



def calcuate_R_square(data_frame, xColmn, yColmn):

    """
    this function will calculate the R-squared value between two variables of a data frame.
    """

    if len(data_frame) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(data_frame[xColmn], data_frame[yColmn])
        return r_value ** 2
    return None

"""1. this function will get a data frame and list of contries, and will create for each counrty illustration that his X-axis is 'Energy Supply per Capita' and the Y-axis is 'GDP'."""

def energy_gdp_illustration(data_frame, contry_list):

    """
    at this function will create scatter plots of 'Energy Supply per Capita' vs 'GDP' in multiple countries.
    """

    final_data_frame = edit_the_data(data_frame)
    row = calc_array_square(contry_list)
    col = row


    # createing the plots, and convert the axis's to 1-dimensional array
    fig, axs = plt.subplots(row, col, figsize=(15, 5 * row), sharex=True, sharey=True)
    axs = axs.flatten()


    # setting a title for thr graph.
    plt.suptitle('Energy Supply per Capita vs GDP in Different Years', fontsize=15)



    # Define color map of the 'TIME' column.
    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=data_frame['TIME'].min(), vmax=data_frame['TIME'].max())


    for i, area in enumerate(contry_list):
        # get the data of each country in the list.
        area_data = final_data_frame[final_data_frame['LOCATION'] == area]


        scatter = plot_scatter_create(axs[i], 'Energy_Supply_per_Capita', 'Value', area_data, cmap, norm)
        set_the_plot_titles(axs[i], f"Country Code : {area}", 'Energy Supply per Capita', 'GDP')

        years = area_data['TIME'].unique()


        # adding the stare and the end year of data of area.
        if len(years) > 0:
            min_year, max_year = min(years), max(years)
            add_years_to_plot(axs[i], min_year, max_year)

        # ensuare the x-axis values will be in all the plots.
        axs[i].tick_params(axis='x', labelbottom=True)


       # will add a color map th the plot.
        add_colormap_to_plot(fig, axs[i], cmap, norm)


    fig.subplots_adjust(top=0.9, left=0.2, wspace=0.4, hspace=0.5)

    plt.show()

"""2. this function will get a data frame and list of contries, and will create for each counrty illustration that his X-axis is 'Energy Supply per Capita' and the Y-axis is 'GDP', and will add a regression line for each plot."""

def energy_gdp_illustration_with_regression(data_frame, contry_list):

    """
    at this function will create scatter plots of 'Energy Supply per Capita' vs 'GDP' in multiple countries with regression line.
    """

    # convert the 'TIME' column
    final_data_frame = edit_the_data(data_frame)
    row = calc_array_square(contry_list)
    col = row



    # createing the plots, and convert the axis's to 1-dimensional array
    fig, axs = plt.subplots(row, col, figsize=(15, 5 * row), sharex=True, sharey=True)
    axs = axs.flatten()


    # setting a title for thr graph.
    plt.suptitle('Energy Supply per Capita vs GDP in Different Years With Regression Line', fontsize=15)


    # Define color map of the 'TIME' column.
    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=data_frame['TIME'].min(), vmax=data_frame['TIME'].max())


    for i, area in enumerate(contry_list):

        # get copy of the data of each country in the list.
        single_country_data = data_frame[data_frame['LOCATION'] == area].copy()
        single_country_data = clean_the_data(single_country_data, ['Energy_Supply_per_Capita', 'Value'])


        # create a plot, ans set his title and add a regression line.
        plot = plot_scatter_create(axs[i], 'Energy_Supply_per_Capita', 'Value', single_country_data, cmap, norm)
        add_regression_line(axs[i], 'Energy_Supply_per_Capita', 'Value', single_country_data)

        # set the plot title, and add the R**2 val, if not is none.
        set_the_plot_titles(axs[i], f" {area} (R² ={calcuate_R_square(single_country_data, 'Energy_Supply_per_Capita', 'Value'):.2f})", 'Energy Supply per Capita', 'GDP')


        years = single_country_data['TIME'].unique()

        if len(years) > 0:
            min_year, max_year = min(years), max(years)
            add_years_to_plot(axs[i], min_year, max_year)


        # ensuare the x-axis values will be in all the plots.
        axs[i].tick_params(axis='x', labelbottom=True)


       # will add a color map th the plot.
        add_colormap_to_plot(fig, axs[i], cmap, norm)


    fig.subplots_adjust(top=0.9, left=0.2, wspace=0.4, hspace=0.5)
    plt.show()

"""this function will use it in Question 3, to clean the data from outliers."""

def clean_data_from_outliers(data_frame, columns):

    """
    this function will clean the data from outliers in the specified columns.
    """

    # Calculate the IQR for the specified columns
    firstQ = data_frame[columns].quantile(0.25)
    secondQ = data_frame[columns].quantile(0.75)
    distanse = secondQ - firstQ



    # Define outlier conditions based on IQR
    outlier_condition = (
        (data_frame[columns[0]] < (firstQ[columns[0]] - 1.5 * distanse[columns[0]])) |
        (data_frame[columns[0]] > (secondQ[columns[0]] + 1.5 * distanse[columns[0]])) |
        (data_frame[columns[1]] < (firstQ[columns[1]] - 1.5 * distanse[columns[1]])) |
        (data_frame[columns[1]] > (secondQ[columns[1]] + 1.5 * distanse[columns[1]]))
    )


    # get outliers and print them.
    outliers = data_frame[outlier_condition]
    if not outliers.empty:
        print("Outliers detected:")
        print(outliers)


    # Remove outliers from the data
    data_frame_without_outliers = data_frame[~outlier_condition]

    return data_frame_without_outliers

"""3. in this question will create a functuion that gets a dataframe, contreis list and year's list, and for each year will create plot of the countries in this year."""

def energy_gdp_illustration_by_year(data_frame, contry_list, year_list):

  """
  at this function creates scatter plots for the relationship between 'Energy Supply per Capita' and 'GDP' for different years
  and also will add a regression line for each plot.
  """


  # each country will get a spicfic color.
  color_map = mpl.colormaps['tab20']
  region_colors = {region: color_map(i) for i, region in enumerate(contry_list)}


  # calcuate the square of the years array.
  row = calc_array_square(year_list)
  col = row


  # sort the year list to chronological order, and add a dictionary for the outliers.
  year_list.sort()
  outliers = {}


   # createing the plots, and convert the axis's to 1-dimensional array
  fig, axs = plt.subplots(row, col, figsize=(15, 5 * row), sharex=True, sharey=True)
  axs = axs.flatten()

  # convert the 'TIME' column to numric.
  data_frame = edit_the_data(data_frame)


  for i, year in enumerate(year_list):

        # get the data for each year to data frame, and leave thier just the contries there are in contry_list.
        filtered_data = data_frame[data_frame['TIME'] == year]
        filtered_data = filtered_data[filtered_data['LOCATION'].isin(contry_list)].copy()


        # convert the we had get to numirc.
        filtered_data = clean_the_data(filtered_data, ['Energy_Supply_per_Capita', 'Value'])


        # cleaning the from outliers with '' function.
        cleaned_data = clean_data_from_outliers(filtered_data, ['Energy_Supply_per_Capita', 'Value'])


        for contry in contry_list:
            region_data = filtered_data[filtered_data['LOCATION'] == contry]
            axs[i].scatter(region_data['Energy_Supply_per_Capita'], region_data['Value'], color=region_colors[contry],  label=contry if i == 0 else "")


        # add a regression line to each plot.
        add_regression_line(axs[i], 'Energy_Supply_per_Capita', 'Value', filtered_data)


        # set the plot title, and add the R**2 val, if not is none.
        set_the_plot_titles(axs[i], f" Year: {year}  (R² ={calcuate_R_square(cleaned_data, 'Energy_Supply_per_Capita', 'Value'):.2f})", 'Energy Supply per Capita', 'GDP')


  handles, labels = [], []
  for region, color in region_colors.items():
            handle = plt.Line2D([0], [0], marker='o', color='w', label=region, markerfacecolor=color, markersize=8)
            handles.append(handle)
            labels.append(region)



  fig.legend(handles=handles, labels=labels, loc='upper center',  ncol=3,  fontsize=10, bbox_to_anchor=(0.5, 1.03))

   # Adjust layout and margins

  plt.show()

"""4. this function will get a year a dataframe, and bulid a two plot's that displaying histograms of the correlation values between energy per capita and GDP per capita."""

def energy_gdp_histogram_plots(final_data_frame, year, bins = 10):


       # convert the values of column's 'TIME', 'Energy_Supply_per_Capita' and 'Value' to numeric, and handling errors by using 'errors = coerce'.
       final_data_frame['TIME'] = pd.to_numeric(final_data_frame['TIME'], errors='coerce')
       final_data_frame['Energy_Supply_per_Capita'] = pd.to_numeric(final_data_frame['Energy_Supply_per_Capita'], errors='coerce')
       final_data_frame['Value'] = pd.to_numeric(final_data_frame['Value'], errors='coerce')


       # distribution the data to 2 array, 1 for all the year's committee the spicfic year(non-include), from the parmter year to last year in the data.
       before_year = final_data_frame[final_data_frame['TIME'] < year]
       after_year = final_data_frame[final_data_frame['TIME'] >= year]


       # calcuate the correlation of the two arrays.
       correlation_before = calculate_correlations(before_year)
       correlation_after = calculate_correlations(after_year)


       # create the plots
       fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey = True)

       # adding the data to plots.
       axs[0].hist(correlation_before, bins=bins, color='blue', alpha=0.7)
       axs[1].hist(correlation_after, bins=bins, color='red', alpha=0.7)

       # set a title for each plot.
       axs[0].set_title(f'Correlation Distribution Before Year {year} (Not-include)')
       axs[1].set_title(f'Correlation Distribution After Year {year} (include)')


       # seting a y and x label for the plots.
       axs[0].set_xlabel('Correlation')
       axs[1].set_xlabel('Correlation')
       axs[0].set_ylabel('Number of Locations')

       #add a dashed grid with transparency to the y-axis of the first two plots to improve reading it.
       axs[0].grid(axis='y', linestyle='--', alpha=0.7)
       axs[1].grid(axis='y', linestyle='--', alpha=0.7)

       # title for graph of the plots.
       fig.suptitle(f"Correlation Histograms of Energy Supply per Capita vs GDP per Capita (Split at the {year})", fontsize=15)

       plt.tight_layout()

       plt.show()

"""5. this function will get as a parmters data frame, and contry list, will cacluate the energy efficiency for each country."""

def energy_efficiency(final_data_frame, contry_list):


      # calcuate the energy efficiency for the data frame.
      final_data_frame['Energy_Efficiency'] = final_data_frame['Value'] / final_data_frame['Energy_Supply_per_Capita']


      # filter the data frame, and geting the data of the contries we had got in the 'contry_list' parmter.
      energy_efficiency_filterd_data = final_data_frame[final_data_frame['LOCATION'].isin(contry_list)]


      # gruop the data by 'LOCATION' and 'TIME' and calcute the mean.
      energy_efficiency_data = energy_efficiency_filterd_data.groupby(['LOCATION', 'TIME'])['Energy_Efficiency'].mean().reset_index()


      plt.figure(figsize=(15, 8))
      for country in contry_list:
            country_data = energy_efficiency_data[energy_efficiency_data['LOCATION'] == country]
            plt.plot(country_data['TIME'], country_data['Energy_Efficiency'], marker='o', label=country)


      # adding title and lebals for the plot.
      plt.title('Energy Efficiency Over Time for Selected Countries')
      plt.xlabel('Year')
      plt.ylabel('Energy Efficiency')

      # adding a legend for the plot, and grid.
      plt.legend(title="Regions", fontsize=12, loc="upper left")
      plt.grid(True)

      plt.show()

"""Main Code"""

def main():

      countires = ['AUS', 'CAN', 'FIN', 'IND', 'ISL', 'ISR', 'RUS', 'SAU', 'SGP', 'USA', 'ZAF', 'ZMB']
      Year = [1990, 1993, 2003, 2005, 1986, 1980, 1981, 2001, 1998, 2017, 1977, 2015]

      warnings.filterwarnings("ignore", category=DeprecationWarning)

      energy_gdp_illustration(final_data_frame, countires)
      energy_gdp_illustration_with_regression(final_data_frame, countires)


      energy_gdp_illustration_by_year(final_data_frame, countires,Year)


      energy_gdp_histogram_plots(final_data_frame,1990)
      energy_efficiency(final_data_frame, countires)


main()
