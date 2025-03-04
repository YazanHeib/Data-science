# -*- coding: utf-8 -*-


from google.colab import files
upload_csv_files = files.upload()

"""Question 2 : uplaod the three file had been uploaded to dataFrames"""

import pandas as pd

electricity_generation_file = pd.read_csv("electricity_generation.csv")

primary_energy_supply_file = pd.read_csv("primary_energy_supply.csv")

renewable_energy_file = pd.read_csv("renewable_energy.csv")

# conversion from MLN_TOE to GWH.
from_MLNTOE_to_GWH = 11630

# conversion from KTOE to GWH.
from_KTOE_to_GWH = 11.63

"""Qustion 3 : create a variable that will contain the name of a region"""

region = "ISR"

"""Question 4"""

#  Select only the rows where LOCATION equals the name of the region.
region_electric_supply = primary_energy_supply_file[primary_energy_supply_file['LOCATION'] == region]



# select only the rows where 'MEASURE' equal to 'MLN_TOE', choose it from the spicifc rows of region.
measure_electric_supply = region_electric_supply[region_electric_supply['MEASURE'] == 'MLN_TOE']



# omit the rows that 'Value' is 'Na'.
real_electric_supply = measure_electric_supply.dropna(subset=['Value'])



# Add a new column of the value in 'GWH' Measure.
real_electric_supply = real_electric_supply.assign(**{'Value_in_GWH': real_electric_supply['Value'] * from_MLNTOE_to_GWH})

"""Qustion 5"""

#  Select only the rows where LOCATION equals the name of the region.
region_electric_generation = electricity_generation_file[electricity_generation_file['LOCATION'] == region]



# select only the rows that the subject equals to 'TOT'.
electric_generation = region_electric_generation[region_electric_generation['SUBJECT'] == 'TOT']



# omit the rows that 'Value' is 'Na'.
total_electric_generation_region = electric_generation.dropna(subset=['Value'])

"""Qustion 6"""

# Select only the rows where LOCATION equals the name of the region.
renewable_energy = renewable_energy_file[renewable_energy_file['LOCATION'] == region]



# select all the rows that had ToT 'Total electricity_generation'
total_renewable_energy = renewable_energy[renewable_energy['SUBJECT'] == 'TOT']



#select the rows that the measure is 'KTOE'
all_renewable_energy_KTOE = total_renewable_energy[total_renewable_energy['MEASURE'] == 'KTOE']



# Add a new column of the value in 'GWH' Measure.
all_renewable_energy_KTOE = all_renewable_energy_KTOE .assign(**{"renewable Value in GWH": all_renewable_energy_KTOE ['Value'] * from_KTOE_to_GWH})

"""Question 7"""

# create a new data frame, that will contain all the electric in GWH.
electric_in_GWH = all_renewable_energy_KTOE[["renewable Value in GWH"]].set_index(all_renewable_energy_KTOE["TIME"])



#adding the supply energy to the data frame.
electric_in_GWH["energy supply in GWH"] = real_electric_supply[["Value_in_GWH"]].set_index(real_electric_supply['TIME'])



#adding the value of electric generiton to data frame.
electric_in_GWH = pd.merge( electric_in_GWH,  total_electric_generation_region[["TIME", "Value"]],   on="TIME",   how="left" )

electric_in_GWH.rename(columns={"Value": "electric generation in GWH"}, inplace=True)


# We'll define a list of the columns we want to check and remove them if they contain missing data.
coulms_to_delete_years = ["renewable Value in GWH","energy supply in GWH", "electric generation in GWH"]



# delete all the years that we don't have any information about in the checks.
electric_in_GWH = electric_in_GWH.dropna(subset=coulms_to_delete_years,how = "all")


electric_in_GWH.set_index('TIME', inplace=True)

"""Summary of the information"""

# to calculte the average annual change magnitude
# well create a new data frame and calculate the difference from year to year by 'diff' method.

change_Magnitude_data_frame = pd.DataFrame()

change_Magnitude_data_frame['renewable Value in GWH'] = electric_in_GWH['renewable Value in GWH'].diff()
change_Magnitude_data_frame['energy supply in GWH'] = electric_in_GWH['energy supply in GWH'].diff()
change_Magnitude_data_frame['electric generation in GWH'] = electric_in_GWH['electric generation in GWH'].diff()

#calucate the avaregre by 'mean' method.
avr_change_renewable = change_Magnitude_data_frame['renewable Value in GWH'].mean()
avr_change_supply = change_Magnitude_data_frame['energy supply in GWH'].mean()
avr_change_generation = change_Magnitude_data_frame['electric generation in GWH'].mean()


print(f"Average annual change for renewable energy value in GWH : {avr_change_renewable:.3f}")
print(f"Average annual change for energy supply in GWH : {avr_change_supply:.3f}")
print(f"Average annual change for total value in GWH : {avr_change_generation:.3f}")

import matplotlib.pyplot as plt


# bulid a graph in the size 18x10.
plt.figure(figsize=(23,11))


# set the x axis to the years thar we have information about.
plt.xticks(ticks=electric_in_GWH.index, labels=electric_in_GWH.index, rotation=90, fontsize=9)


# set the y axis the energy supply.
plt.yticks(list(range(0,340001,10000)))


plt.plot(electric_in_GWH.index, electric_in_GWH['renewable Value in GWH'], marker='o', label='renewable Value in GWH', color = 'limegreen')
plt.plot(electric_in_GWH.index, electric_in_GWH['electric generation in GWH'], marker='o', label='electric generation in GWH', color = 'blue')
plt.plot(electric_in_GWH.index, electric_in_GWH['energy supply in GWH'], marker='o', label='energy supply in GWH', color  = 'fuchsia')


# Design the graph.
plt.title(f'\nEnergy Supply in {region} Over the Years\n', fontsize=15)
plt.xlabel('Year', fontsize=13)
plt.ylabel('Energy Supply (GWH)', fontsize=13)
plt.legend(loc='upper left')

plt.show()

"""Information exploration"""

# print all the years that the three metrics is in negative tendency.
negative_tendency = change_Magnitude_data_frame[(change_Magnitude_data_frame['renewable Value in GWH'] < 0) & (change_Magnitude_data_frame['energy supply in GWH'])
 & (change_Magnitude_data_frame['electric generation in GWH'] < 0)]



# to calculate the percentage of renewable energy out of the total energy.
# sum the values for all years by 'sum' method, divide by the total energy production, and multiply by 100.



the_total_renewable_energy = electric_in_GWH['renewable Value in GWH'].sum()
the_total_supply = electric_in_GWH["energy supply in GWH"].sum()

renewable_percentage = (the_total_renewable_energy / the_total_supply) * 100


print("the Year's in negative tendency is :",list(negative_tendency.index))
print(f"the percentage of renewable energy :{renewable_percentage:.3f} %")
