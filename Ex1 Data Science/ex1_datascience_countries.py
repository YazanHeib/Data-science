# -*- coding: utf-8 -*-


from google.colab import files
upload_csv_files = files.upload()

"""uplaod the three file had been uploaded to dataFrames"""

import pandas as pd

electricity_generation_file = pd.read_csv("electricity_generation.csv")

primary_energy_supply_file = pd.read_csv("primary_energy_supply.csv")

renewable_energy_file = pd.read_csv("renewable_energy.csv")

# conversion from MLN_TOE to GWH.
from_MLNTOE_to_GWH = 11630

# conversion from KTOE to GWH.
from_KTOE_to_GWH = 11.63

""" create a list that will contain the name of a 5 Countries"""

regions = ["ISR","AUS","BEL","CAN","CZE"]

"""this function will read the primary_energy_supply file and return a datefeame of the data."""

def read_primary_energy_supply(region):

           region_electric_supply = primary_energy_supply_file[primary_energy_supply_file['LOCATION'] == region]



           # select only the rows where 'MEASURE' equal to 'MLN_TOE', choose it from the spicifc rows of region.
           measure_electric_supply = region_electric_supply[region_electric_supply['MEASURE'] == 'MLN_TOE']



           # omit the rows that 'Value' is 'Na'.
           real_electric_supply = measure_electric_supply.dropna(subset=['Value'])



           # Add a new column of the value in 'GWH' Measure.
           real_electric_supply = real_electric_supply.assign(**{'Value_in_GWH': real_electric_supply['Value'] * from_MLNTOE_to_GWH})


           return real_electric_supply

"""this function will get a region and read the electricity_generation file, return dataframe of his data."""

def read_electricity_generation(region):


    #  Select only the rows where LOCATION equals the name of the region.
    region_electric_generation = electricity_generation_file[electricity_generation_file['LOCATION'] == region]

    # select only the rows that the subject equals to 'TOT'.

    electric_generation = region_electric_generation[region_electric_generation['SUBJECT'] == 'TOT']

    # omit the rows that 'Value' is 'Na'.
    total_electric_generation_region = electric_generation.dropna(subset=['Value'])


    return total_electric_generation_region

"""this function will read the renewable_energy file, by spicfic region and return a dataframe"""

def read_renewable_energy(region):


    # Select only the rows where LOCATION equals the name of the region.
    renewable_energy = renewable_energy_file[renewable_energy_file['LOCATION'] == region]

    # select all the rows that had ToT 'Total electricity_generation'
    total_renewable_energy = renewable_energy[renewable_energy['SUBJECT'] == 'TOT']

    # select the rows that the measure is 'KTOE'
    all_renewable_energy_KTOE = total_renewable_energy[total_renewable_energy['MEASURE'] == 'KTOE']

    # Add a new column of the value in 'GWH' Measure.
    all_renewable_energy_KTOE = all_renewable_energy_KTOE.assign(
    **{"renewable Value in GWH": all_renewable_energy_KTOE['Value'] * from_KTOE_to_GWH})


    return all_renewable_energy_KTOE

"""this function will delete all the year's we don't have a informatiom about, and return a new data frame of all the electic in GWH"""

def clean_data(total_electric_generation_region,real_electric_supply,all_renewable_energy_KTOE):


    # create a new data frame, that will contain all the electric in GWH.
    electric_in_GWH = all_renewable_energy_KTOE[["renewable Value in GWH"]].set_index(all_renewable_energy_KTOE["TIME"])

    # adding the supply energy to the data frame.
    electric_in_GWH["energy supply in GWH"] = real_electric_supply[["Value_in_GWH"]].set_index(real_electric_supply['TIME'])

    # adding the value of electric generiton to data frame.
    electric_in_GWH = pd.merge(electric_in_GWH, total_electric_generation_region[["TIME", "Value"]], on="TIME", how="left")

    electric_in_GWH.rename(columns={"Value": "electric generation in GWH"}, inplace=True)

    # We'll define a list of the columns we want to check and remove them if they contain missing data.
    coulms_to_delete_years = ["renewable Value in GWH", "energy supply in GWH", "electric generation in GWH"]

    # delete all the years that we don't have any information about in the checks.
    electric_in_GWH = electric_in_GWH.dropna(subset=coulms_to_delete_years, how="all")

    electric_in_GWH.set_index('TIME', inplace=True)

    return electric_in_GWH

"""this function will calucte the Negative tendency years and the Growth of renewable Energy also the Total Energy Supply in GWH."""

def calculate_the_data(countries_data_frame, electric_in_GWH, region):


        change_Magnitude_data_frame = pd.DataFrame()

        # a new data frame that have the difference between the year's.
        change_Magnitude_data_frame['renewable Value in GWH'] = electric_in_GWH['renewable Value in GWH'].diff()
        change_Magnitude_data_frame['energy supply in GWH'] = electric_in_GWH['energy supply in GWH'].diff()
        change_Magnitude_data_frame['electric generation in GWH'] = electric_in_GWH['electric generation in GWH'].diff()


        #calcute all the years that the three metrics is in negative tendency.
        negative_tendency_index = change_Magnitude_data_frame[(change_Magnitude_data_frame['renewable Value in GWH'] < 0) & (change_Magnitude_data_frame['energy supply in GWH']) &
                                                        (change_Magnitude_data_frame['electric generation in GWH'] < 0)].index


        negative_tendency = negative_tendency_index.tolist()



        # calaute renewable energy as a percentage of total energy supply.
        renewable_per = (((electric_in_GWH['renewable Value in GWH'].iloc[-1]) - (electric_in_GWH['renewable Value in GWH'].iloc[0])) /  (electric_in_GWH['renewable Value in GWH'].iloc[0]))* 100


        total_energy = electric_in_GWH['energy supply in GWH'].sum()


        new_country = {'Region' : [region], 'Negative tendency years' : [negative_tendency], 'Growth of renewable Energy' : [renewable_per], 'Total Energy Supply in GWH' : [total_energy ] }



        new_country_data_frame = pd.DataFrame(new_country).dropna()
        countries_data_frame = countries_data_frame.dropna()
        new_country_data_frame.set_index('Region', inplace=True)


        # Use pd.concat to append the new row to the DataFrame
        countries_data_frame = pd.concat([countries_data_frame, new_country_data_frame])


        return countries_data_frame.dropna()

"""main function"""

from IPython.display import display

def main():



       #new data frame of the 5 countries.
      columns_name = ['Region', 'Negative tendency years', 'Growth of renewable Energy', 'Total Energy Supply in GWH']
      countries_data_frame = pd.DataFrame(columns=columns_name)
      countries_data_frame.set_index('Region', inplace=True)



      for region in regions:

           total_electric_generation_region = read_electricity_generation(region)
           real_electric_supply = read_primary_energy_supply(region)
           all_renewable_energy_KTOE = read_renewable_energy(region)



           electric_in_GWH = clean_data(total_electric_generation_region,real_electric_supply,all_renewable_energy_KTOE)

           countries_data_frame = calculate_the_data(countries_data_frame, electric_in_GWH, region)



      display(countries_data_frame)

main()
