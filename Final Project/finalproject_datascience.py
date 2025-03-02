# -*- coding: utf-8 -*-

from google.colab import files
data_uploaded = files.upload()

"""1. Upload The Data To Data Frames.
2. Import The Libraries Will use In The Code.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import math
import warnings
import gc
import numpy as np
from scipy.stats import linregress
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from matplotlib.ticker import MaxNLocator



# Upload CSV Files To Data Frames.
daily_income_data_frame = pd.read_csv('mincpcap_cppp.csv')

carbon_chronicles_data_frame = pd.read_csv('co2_pcap_cons.csv')

gdp_data_frame = pd.read_csv('gdp_pcap.csv')

lifeSpan_legder_data_frame = pd.read_csv('lex.csv')

"""this is the function's that will use in the code."""

def clean_data_frame(data):
    """
     This Function Will Clear A Countries With Missing Values, And Keeping Only That Countreis With Complete Data.
    """
    return data.dropna(axis=0)


def convert_K_values(data_frame):
    """
    This function converts values containing 'K' (e.g., '29.4K') into numerical values (e.g., 29400).
    """
    for column in data_frame.columns[2:]:
       data_frame[column] = (data_frame[column].astype(str) .str.replace(r'K|k', '000', regex=True).str.replace('−', '-', regex=True).replace({',': ''}, regex=True) .astype(float))

    return data_frame



def convert_To_Float_32(data_frame):
    """
    This function converts a DataFrame Numric Columns To Float 32.
    To Save The Data That The Ram Will Use.
    """
    numeric_columns = data_frame.select_dtypes(include=['float64']).columns
    data_frame[numeric_columns] = data_frame[numeric_columns].astype('float32')

    return data_frame

"""clean All The Missing Data From The Data Frame."""

# cleaning All The DataFrames From Missing Values.
daily_income_data_frame = clean_data_frame(daily_income_data_frame)
carbon_chronicles_data_frame = clean_data_frame(carbon_chronicles_data_frame)
gdp_data_frame = clean_data_frame(gdp_data_frame)
lifeSpan_legder_data_frame = clean_data_frame(lifeSpan_legder_data_frame)

"""1. Select 40 Countries To Examine And Analyze The Data.
2. Set The Year's Range.
"""

countries = [
    "USA", "Germany", "Japan", "South Korea", "Singapore", "Australia", "Israel", "Ireland", "Norway", "Canada",
    "China", "India", "Brazil", "Indonesia", "Turkey", "Mexico", "Vietnam", "North Korea", "Luxembourg", "Chile",
    "Russia", "Ukraine", "Venezuela", "Greece", "Argentina", "Lebanon", "Iran", "South Africa", "Egypt", "Cuba",
    "Nigeria", "Ethiopia", "Bangladesh", "Pakistan", "Philippines", "Rwanda", "Colombia", "Saudi Arabia", "Kazakhstan", "Angola"
]

years = list(range(1950, 2025))

"""## Part 1 Of The Data Analayze.

Create A Data Frame That Will Contain This Coulmns In The Year's 1950-2024.
1. Country : The Name Of The Country.
2. Year : The Year Of The Recorded Data.
3. CO2 : (Tonnes per capita) Carbon Dioxide Emissions Per Capita
4. GDP : (International dollars, 2017 PPP) – Gross Domestic Product Per Capita.
5. Life Expectancy : (Years) – The Expected Number Of Years A Newborn Would Live.
6. Income :  (International dollars per day, 2017 PPP) – The mean daily household income per capita.
"""

def Create_Data_Frame_In_Range(countries, years, Co2_df, gdp_df, lex_df, income_df):
    """
    At This Function Will
    """

    data_frame_list = []

    for country in countries:
         for year in years:

             # get This String Val Of Year.
             year_str = str(year)

             # Check If The Data About The Spicfic Year Is In The All Files.
             if year_str in Co2_df.columns and year_str in gdp_df.columns and year_str in lex_df.columns and year_str in income_df.columns:

                   # Get The Data From Data Frame.
                   co2_data = Co2_df.set_index("country").get(year_str, {}).get(country, None)
                   gdp_data = gdp_df.set_index("country").get(year_str, {}).get(country, None)
                   lex_data = lex_df.set_index("country").get(year_str, {}).get(country, None)
                   income_data = income_df.set_index("country").get(year_str, {}).get(country, None)

                   # Add The To The List.
                   data_frame_list.append([country, year, co2_data, gdp_data, lex_data, income_data])

    # Create A New Data Frame, That Will Contain The Spicfic Columns.
    merge_data_frame = pd.DataFrame(data_frame_list, columns=["Country", "Year", "CO2", "Daily Income","GDP","Life Expectancy"])
    return merge_data_frame

"""Call The Function We Create."""

merge_data_frame = Create_Data_Frame_In_Range(countries, years, carbon_chronicles_data_frame, gdp_data_frame, lifeSpan_legder_data_frame, daily_income_data_frame)

"""1. Convert All The Coulmns That Contain Number's In The Format of '##K' To Real Number.
2. Convert All THe Columns THat Contain Number Ti float 32.
"""

# Call THe Function Will Convert The Data.
merge_data_frame = convert_K_values(merge_data_frame)
merge_data_frame = convert_To_Float_32(merge_data_frame)
merge_data_frame.to_csv('merge_data_frame.csv', index=False)

"""Will Start With Calcuate The Medain, Standard Deviation And Avarege."""

# Get The Numric Columns.
numeric_cols = merge_data_frame.select_dtypes(include=['number']).drop(columns=['Year'], errors='ignore')

# Caluate The Median, Standard Deviation And Avarege By 'describe()'.
stats = numeric_cols.describe().loc[['mean', 'std']]
stats.loc['median'] = numeric_cols.median()

print(stats)

"""Calcuate The Correlation Between GDP AND Life Expectancy"""

# Calacuate The Correaltion Betweem The GDP And Life Expectancy.
first_corr = merge_data_frame['GDP'].corr(merge_data_frame['Life Expectancy'])


# Calacuate The Correaltion Betweem The CO2 And Life Expectancy.
second_corr = merge_data_frame['CO2'].corr(merge_data_frame['Life Expectancy'])


# Calacuate The Correaltion Betweem The Daily Income And Life Expectancy.
third_corr = merge_data_frame['Daily Income'].corr(merge_data_frame['Life Expectancy'])


print(f"The Correlation Between GDP And Life Expectancy Is :  {first_corr:.4f}")
print(f"The Correlation Between CO2 And Life Expectancy Is :  {second_corr:.4f}")
print(f"The Correlation Between Daily Income And Life Expectancy Is : {third_corr:.4f}")

"""This Function That Will Create A Plot Between Two Coulmns Of The Data Frame."""

def plot_relationship_to_lex(data_frame, x_column, x_label, color='b'):
  """
  At This Function Will Create A Plot Between Life Expectancy And Other Varibels Of The Data Frame.
  """

  # Create A Plot.
  plt.figure(figsize=(10, 6))

  # Add The Data To The Data Frame.
  plt.scatter(data_frame[x_column], data_frame["Life Expectancy"], alpha=0.5, color=color, label="Data")


  # Calculate the linear regression line
  slope, intercept, r_value, p_value, std_err = linregress(data_frame[x_column], data_frame["Life Expectancy"])
  regression_line = slope * data_frame[x_column] + intercept

  # Add To The Plot.
  plt.plot(data_frame[x_column], regression_line, color='black', label=f'Linear Regression (R={r_value:.2f})')


  # Set The Title, X-Axis And Y-Axis Label.
  plt.title(f"{x_label} vs Life Expectancy")
  plt.xlabel(x_label)
  plt.ylabel("Life Expectancy")


  # Calcuate And Add The Stat Text.
  stats_text = f"R² = {r_value**2:.2f}\nP-value = {p_value:.2g}\nStd Error = {std_err:.2f}"
  plt.text(0.05, 0.85, stats_text, transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))



  # Show The Plot.
  plt.legend()
  plt.grid()
  plt.show()

"""plot That Will Show The Relationship Between GDP And Life Expectancy With Regression Line."""

plot_relationship_to_lex(merge_data_frame, 'GDP', 'GDP per Capita', color='b')

"""plot That Will Show The Relationship Between CO2 And Life Expectancy With Regression Line."""

plot_relationship_to_lex(merge_data_frame, 'CO2', 'CO2 Emissions', color='r')

"""plot That Will Show The Relationship Between Daily Income And Life Expectancy With Regression Line."""

plot_relationship_to_lex(merge_data_frame, 'Daily Income', 'Daily Income', color='g')

"""graph That Will Show Trend Over Time (Average Across All Countries)."""

# Create A plot.
plt.figure(figsize=(14, 7))


# Calcuate The Avagege For each Year.
year_avg_data = merge_data_frame.groupby("Year")[["GDP", "CO2", "Life Expectancy", "Daily Income"]].mean()



# Add The Data To The Graph.
plt.plot(year_avg_data.index, year_avg_data["GDP"], label="GDP", color='b')
plt.plot(year_avg_data.index, year_avg_data["CO2"], label="CO2", color='r')
plt.plot(year_avg_data.index, year_avg_data["Life Expectancy"], label="Life Expectancy", color='g')
plt.plot(year_avg_data.index, year_avg_data["Daily Income"], label="Daily Income", color='y')


# Set The Title, X-Axis And Y-Axis Label.
plt.title("Trends of Key Indicators Over Time", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Values (Standardized Scale)", fontsize=12)

# Show The Graph.
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

"""Build A Linear Regression Model To Predict Life Expectancy."""

# X is Explanatory Variables And Y is Target Variable.
X = merge_data_frame[["GDP", "CO2", "Daily Income"]]
y = merge_data_frame["Life Expectancy"]


# Splitting The Data Into A Training Set (75%) And A Test Set (25%).
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Create A Linear Regersion Model.
regersion_model = LinearRegression()


# Train The Model.
regersion_model.fit(x_train, y_train)


# Make Predictions On The Test Set.
y_pred = regersion_model.predict(x_test)


# Model Performance Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mse)


# Print The Evaluation Metrics.
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

"""Build A Model Of To Random Forest Predict Life Expectancy."""

# X is Explanatory Variables And Y is Target Variable.
X = merge_data_frame[["GDP", "CO2", "Daily Income"]]
y = merge_data_frame["Life Expectancy"]


# Splitting The Data Into A Training Set (85%) And A Test Set (20%).
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# Create A Random Forset Model.
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)


# Train The Model.
random_forest_model.fit(x_train, y_train)


# Make Predictions On The Test Set.
y_pred = random_forest_model.predict(x_test)


# Model Performance Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)


# Print The Evaluation Metrics.
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

"""# Part B : How do GDP per capita, daily income, and CO₂ emissions affect life expectancy in different populations around the world in the 19th and 20th centuries?

Clear All The Countreis That We Will Not Check.
"""

def filter_countries(data_frame):
  """
  At This Function Will Clear All The Countreis That We Will Not Check.
  """
  return data_frame[data_frame.iloc[:, 0].isin(countries)]

# Call THe Function For The Data Frames.
daily_income_data_frame = filter_countries(daily_income_data_frame)
carbon_chronicles_data_frame = filter_countries(carbon_chronicles_data_frame)
gdp_data_frame = filter_countries(gdp_data_frame)
lifeSpan_legder_data_frame = filter_countries(lifeSpan_legder_data_frame)

"""Clean The Data, And Fill The Missing Data."""

# Convert All The Cells The Contain's Data In Format Of 'K' (29.3K T0 29300).
daily_income_data_frame = convert_K_values(daily_income_data_frame)
carbon_chronicles_data_frame = convert_K_values(carbon_chronicles_data_frame)
gdp_data_frame = convert_K_values(gdp_data_frame)
lifeSpan_legder_data_frame = convert_K_values(lifeSpan_legder_data_frame)

"""At This Function Will Fill Cells Without Data With The Avg Of This Counrty."""

def filling_missing_data(data_frame):
    """
    At This Function Will Fill Cells Without Data With The ((Data Of Previous year + Data Of Next Year) / 2 ) For This Counrty.
    """
    for country in data_frame['country'].unique():
       data_frame.iloc[:, 1:] = data_frame.iloc[:, 1:].interpolate(method='linear', limit_direction='both')

    return data_frame


# Call The Functyion With The Data Frames.
daily_income_data_frame = filling_missing_data(daily_income_data_frame)
carbon_chronicles_data_frame = filling_missing_data(carbon_chronicles_data_frame)
gdp_data_frame = filling_missing_data(gdp_data_frame)
lifeSpan_legder_data_frame = filling_missing_data(lifeSpan_legder_data_frame)

"""Delete The Other Year From The Data Frame."""

# Filter the data to keep only the years between 1800 and 2000
carbon_chronicles_data_frame = carbon_chronicles_data_frame.loc[:, 'country': '2000']
daily_income_data_frame = daily_income_data_frame.loc[:, 'country': '2000']
gdp_data_frame = gdp_data_frame.loc[:, 'country': '2000']
lifeSpan_legder_data_frame = lifeSpan_legder_data_frame.loc[:, 'country': '2000']

"""This Function Will Calcuate The Median, Std And Avarge For Two Data Frames."""

def calculate_statistics(data_frame_19_centenary, data_frame_20_centenary, string):
    """
    At This Function Will Calcuate The Median, Std And Avarge For Two Data Frames.
    """

    numeric_columns_19 = data_frame_19_centenary.select_dtypes(include=np.number)
    numeric_columns_20 = data_frame_20_centenary.select_dtypes(include=np.number)

    # Calcuate The Staticis For The First Data Frame
    mean_19_centenary = numeric_columns_19.mean().mean()
    median_19_centenary = numeric_columns_19.median().median()
    std_19_centenary = numeric_columns_19.std().std()


    # Calcuate The Staticis For The Second Data Frame
    mean_20_centenary = numeric_columns_20.mean().mean()
    median_20_centenary = numeric_columns_20.median().median()
    std_20_centenary = numeric_columns_20.std().std()


    # Create A Data Frame For The Summary.
    summary_stats = pd.DataFrame({
    'Time': ['1800-1900', '1900-2000'],
    'Mean': [mean_19_centenary, mean_20_centenary],
    'Median': [median_19_centenary, median_20_centenary],
    'Standard Deviation': [std_19_centenary, std_20_centenary]
    })

    print(string,'\n')
    print(summary_stats)

"""This Function Will Take Two Datasets And A String Representing The Data They Contain. It Will Perform The Following:

1.    For The Same 40 Countries, It Will Calculate The Average For Each Year.  
2.    Create Two Graphs, Each Representing Data For Its Respective centenary.



"""

def plot_average_over_years(data_frame_19_centenary, data_frame_20_centenary, string, title):
    """
    Plots the average values over the years for two datasets,
    each representing a different centenary, based on the given data type.
    """

    # Remove non-numeric columns
    numeric_columns_19 = data_frame_19_centenary.select_dtypes(include=np.number)
    numeric_columns_20 = data_frame_20_centenary.select_dtypes(include=np.number)


    # Calcuate The Mean Of The 19 And 20 Centenary.
    mean_19_centenary = numeric_columns_19.mean()
    mean_20_centenary = numeric_columns_20.mean()


    # Create Plot For Two Centenaries.
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))


    # Plot for the 19 Centenary, And Set X-Ticks To Be Every 10 Years.
    ax[0].plot(mean_19_centenary.index, mean_19_centenary.values, marker='o', linestyle='-', color='b')
    ax[0].set_xticklabels([str(year) if year % 10 == 0 else '' for year in range(1800, 1901)], rotation=45)


    # Set Title And Y-Axis, X-Axis Labels
    ax[0].set_title(f'{string} At 1800-1900 Years')
    ax[0].set_xlabel('Year')
    ax[0].set_ylabel('Average Value')

    ax[0].grid(True)


    # Plot for the 20 Centenaries, And Set X-Ticks To Be Every 10 Years.
    ax[1].plot(mean_20_centenary.index, mean_20_centenary.values, marker='o', linestyle='-', color='r')
    ax[1].set_xticklabels([str(year) if year % 10 == 0 else '' for year in range(1900, 2001)], rotation=45)


    # Set Title And Y-Axis, X-Axis Labels
    ax[1].set_title(f'{string} At 1900-2000 Years')
    ax[1].set_xlabel('Year')
    ax[1].set_ylabel('Average Value')

    ax[1].grid(True)


    # Set the main title for the entire plot
    fig.suptitle(title, fontsize=16)

    # Display the plots
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust to make room for the title
    plt.show()

"""1. Calculate The Median, Standard Deviation, And Average For Each Century Separately For Carbon Chronicle.
2. Display The Average For Each Year In A Comparison Graph Between The Two Centuries.
"""

# Split The Year To Two Sets, 1: For Year's (1800 - 1900), 2: (1900 - 2000), These Sets Will Show Carbon Chronicles In Different Centuries.
carbon_chronicles_19_centenary = carbon_chronicles_data_frame.iloc[:, :102]
carbon_chronicles_20_centenary = carbon_chronicles_data_frame.iloc[:, 101:]

warnings.filterwarnings("ignore", category=UserWarning)


# Set Country Columns To The 20 Centenary Data Frame.
carbon_chronicles_20_centenary = pd.concat([carbon_chronicles_19_centenary['country'], carbon_chronicles_20_centenary], axis=1)


# Calcuate The Median, Std And Avarge For Each Centenary.
calculate_statistics(carbon_chronicles_19_centenary, carbon_chronicles_20_centenary, 'Carbon Chronicles Statics For 19 And 20 Centenary')


#  Create A Plot For Carbon Chronicles In Different Centuries.
plot_average_over_years(carbon_chronicles_19_centenary, carbon_chronicles_20_centenary, 'Carbon Chronicles', 'The Average Of Carbon Chronicles In The 19th And 20th Centuries')

"""1. Calculate The Median, Standard Deviation, And Average For Each Century Separately For Daily Income.
2. Display The Average For Each Year In A Comparison Graph Between The Two Centuries.
"""

# Split The Year To Two Sets, 1: For Year's (1800 - 1900), 2: (1900 - 2000), These Sets Will Show Daily Income In Different Centuries.
daily_income_data_19_centenary = daily_income_data_frame.iloc[:, :102]
daily_income_data_20_centenary = daily_income_data_frame.iloc[:, 101:]


# Set Country Columns To The 20 Centenary Data Frame.
daily_income_data_20_centenary = pd.concat([daily_income_data_19_centenary['country'], daily_income_data_20_centenary], axis=1)


# Calcuate The Median, Std And Avarge For Each Centenary.
calculate_statistics(daily_income_data_19_centenary, daily_income_data_20_centenary, 'Daily Income Statics For 19 And 20 Centenary')


# Create A Plot For Daily Income In Different Centuries.
plot_average_over_years(daily_income_data_19_centenary, daily_income_data_20_centenary, 'Daily Income', 'The Average Of Daily Income In The 19th And 20th Centuries')
warnings.filterwarnings("ignore", category=UserWarning)

"""1. Calculate The Median, Standard Deviation, And Average For Each Century Separately For GDP.
2. Display The Average For Each Year In A Comparison Graph Between The Two Centuries.
"""

# Split The Year To Two Sets, 1: For Year's (1800 - 1900), 2: (1900 - 2000), These Sets Will Show GDP In Different Centuries.
gdp_data_19_centenary = gdp_data_frame.iloc[:, :102]
gdp_data_20_centenary = gdp_data_frame.iloc[:, 101:]


# Set Country Columns To The 20 Centenary Data Frame.
gdp_data_20_centenary = pd.concat([gdp_data_19_centenary['country'], gdp_data_20_centenary], axis=1)
gdp_data_19_centenary.to_csv('gdp_data_19_centenary.csv',index = False)
gdp_data_20_centenary.T.to_csv('gdp_data_20_centenary.csv',index = False)


# Calcuate The Median, Std And Avarge For Each Centenary.
calculate_statistics(gdp_data_19_centenary, gdp_data_20_centenary, 'GDP Statics For 19 And 20 Centenary')


# Create A Plot For GDP In Different Centuries.
plot_average_over_years(gdp_data_19_centenary, gdp_data_20_centenary, 'GDP', 'The Average Of GDP In The 19th And 20th Centuries')
warnings.filterwarnings("ignore", category=UserWarning)

"""1. Calculate The Median, Standard Deviation, And Average For Each Century Separately For Life Span Legder.
2. Display The Average For Each Year In A Comparison Graph Between The Two Centuries.
"""

# Split The Year To Two Sets, 1: For Year's (1800 - 1900), 2: (1900 - 2000), These Sets Will Show Life Span Legder In Different Centuries.
lifeSpan_legder_data_19_centenary = lifeSpan_legder_data_frame.iloc[:, :102]
lifeSpan_legder_data_20_centenary = lifeSpan_legder_data_frame.iloc[:, 101:]


# Set Country Columns To The 20 Centenary Data Frame.
lifeSpan_legder_data_20_centenary = pd.concat([lifeSpan_legder_data_19_centenary['country'], lifeSpan_legder_data_20_centenary], axis=1)
lifeSpan_legder_data_20_centenary.to_csv('lifeSpan_legder_data_20_centenary.csv',index = False)

# Calcuate The Median, Std And Avarge For Each Centenary.
calculate_statistics(lifeSpan_legder_data_19_centenary, lifeSpan_legder_data_20_centenary, 'Life Span Legder Statics For 19 And 20 Centenary')


# Create A Plot For Life Span Legder In Different Centuries.
plot_average_over_years(lifeSpan_legder_data_19_centenary, lifeSpan_legder_data_20_centenary, 'Life Span Legder', 'The Average Of Life Span Legder In The 19th And 20th Centuries')
warnings.filterwarnings("ignore", category=UserWarning)

"""Create Databases That Will Contain The Data For 19 Century Separately. These Data Will Help Us Understand What Influenced Life Expectancy In Each Century Individually."""

# Set The Range We Want To Send It To The Function.
centenary_19_range = list(range(1800, 1901))

centenary_19_merge_data_frame = Create_Data_Frame_In_Range(countries, centenary_19_range, carbon_chronicles_19_centenary, daily_income_data_19_centenary, gdp_data_19_centenary, lifeSpan_legder_data_19_centenary)

"""Create Databases That Will Contain The Data For 20 Century Separately. These Data Will Help Us Understand What Influenced Life Expectancy In Each Century Individually."""

# Set The Range We Want To Send It To The Function.
centenary_20_range = list(range(1900, 2001))

centenary_20_merge_data_frame = Create_Data_Frame_In_Range(countries, centenary_20_range, carbon_chronicles_20_centenary, daily_income_data_20_centenary, gdp_data_20_centenary, lifeSpan_legder_data_20_centenary)

"""build a Random Forest Regression model to determine which factor had the greatest impact on life expectancy in the 19th century."""

# X is Explanatory Variables And Y is Target Variable.
X_19 = centenary_19_merge_data_frame[["GDP", "CO2", "Daily Income"]]
y_19 = centenary_19_merge_data_frame["Life Expectancy"]


# Splitting The Data Into A Training Set (80%) And A Test Set (20%).
x_train_19, x_test_19, y_train_19, y_test_19 = train_test_split(X_19, y_19, test_size=0.20, random_state=42)


# Create A Random Forset Model
random_forest_model_19 = RandomForestRegressor(n_estimators=100, random_state=42)


# Train The Model.
random_forest_model_19.fit(x_train_19, y_train_19)


# Make Predictions On The Test Set.
y_pred_19 = random_forest_model_19.predict(x_test_19)



# Model Performence Evaluation.
mae_19 = mean_absolute_error(y_test_19, y_pred_19)
mse_19 = mean_squared_error(y_test_19, y_pred_19)
rmse_19 = math.sqrt(mse_19)
r2_19 = r2_score(y_test_19, y_pred_19)


# Print The Result.
print(f"Mean Absolute Error (MAE): {mae_19:.4f}")
print(f"Mean Squared Error (MSE): {mse_19:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_19:.4f}")
print(f"R-squared (R²): {r2_19:.4f}")

"""build a Random Forest Regression model to determine which factor had the greatest impact on life expectancy in the 20th century."""

# X is Explanatory Variables And Y is Target Variable.
X_20 = centenary_20_merge_data_frame[["GDP", "CO2", "Daily Income"]]
y_20 = centenary_20_merge_data_frame["Life Expectancy"]


# Splitting The Data Into A Training Set (80%) And A Test Set (20%).
x_train_20, x_test_20, y_train_20, y_test_20 = train_test_split(X_20, y_20, test_size=0.2, random_state=42)


# Create A Random Forest Model.
random_forest_model_20 = RandomForestRegressor(n_estimators=100, random_state=42)


# Train The Model.
random_forest_model_20.fit(x_train_20, y_train_20)


# Make Predictions On The Test Set.
y_pred_20 = random_forest_model_20.predict(x_test_20)


# Model Performence Evaluation.
mae_20 = mean_absolute_error(y_test_20, y_pred_20)
mse_20 = mean_squared_error(y_test_20, y_pred_20)
rmse_20 = math.sqrt(mse_20)
r2_20 = r2_score(y_test_20, y_pred_20)


# Print The Result.
print(f"Mean Absolute Error (MAE): {mae_20:.4f}")
print(f"Mean Squared Error (MSE): {mse_20:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_20:.4f}")
print(f"R-squared (R²): {r2_20:.4f}")
