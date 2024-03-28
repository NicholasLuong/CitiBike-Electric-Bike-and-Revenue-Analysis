#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np 

# Folder path 
folder_path = '/Users/benjaminheuberger/Desktop/Berkeley_Spring_2023/DATASCI203/lab-2-lab2-benjamin-phillip-nick/data/citibike_data'  

# Create an empty list to store DataFrames from each file
dataframes_list = []

# Loop through the files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a CSV file
    if filename.endswith('.csv'):
        # Read each CSV file into a DataFrame
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        # Append the DataFrame to the list
        dataframes_list.append(df)

# Concatenate all DataFrames into a single DataFrame
df = pd.concat(dataframes_list, ignore_index=True)

df['started_at'] = pd.to_datetime(df['started_at'])
df['ended_at'] = pd.to_datetime(df['ended_at'])
df['ride_time'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60
 
# Conditions for the ride prices 
condition_member = df['member_casual'] == 'member'
condition_ride_time_casual = df['ride_time'] < 30
condition_ride_time_member = df['ride_time'] < 45
condition_electric_bike = df['rideable_type'] == 'electric_bike'

# Costs 
casual_fee = 4.50 # applies to non-members to start a ride
member_fee = 1.29 # Annual fee divided by the average number of rides/year of members across 2022 -- calculated basedo on Citibike operating data. 

e_rates = {'member': 0.17,
           'casual': 0.26
           }

# Define the ride prices based on the conditions
df['ride_price'] = np.where(
    (condition_member & condition_electric_bike),member_fee + (e_rates['member'] * df['ride_time']),       # Member, electric bike
    np.where((condition_member & condition_ride_time_member), member_fee,                                  # Member, <45 mins, non-electric bike
        np.where((condition_member), member_fee + ((df['ride_time'] - 45) * e_rates['member']),            # Member, >=45 mins, non-electric bike 
            np.where((condition_electric_bike), casual_fee + (e_rates['casual'] * df['ride_time']),        # Casual,  electric bike
                     np.where((condition_electric_bike), casual_fee,                                       # Casual, <30 mins, non-electric bike
                           casual_fee + ((df['ride_time'] - 30) * e_rates['casual'])                       # Casual, >=30 mins, non-electric bike 
                
                )
            )
        )
    )

) 

# Group by station ID and calculate several metrics 
grouped_df = df.groupby('start_station_id').agg({
    'ride_time': 'mean',
    'ride_price': 'mean', 
    'ride_id': 'count', 
    'rideable_type': lambda x: (x == 'electric_bike').mean(), 
    'member_casual': lambda x: (x == 'member').mean(),
    'started_at': ['min', 'max']  
}).reset_index()

# Rename the columns for clarity
grouped_df.columns = ['start_station_id', 
                      'avg_ride_time', 
                      'avg_ride_price', 
                      'total_rides', 
                      'electric_bike_percentage', 
                      'member_percentage', 
                      'min_started_at', 
                      'max_started_at']

grouped_df['days_in_operation'] = (grouped_df['max_started_at'] - grouped_df['min_started_at'])
grouped_df['days_in_operation'] = grouped_df['days_in_operation'].dt.days

del grouped_df['min_started_at'], grouped_df['max_started_at']

grouped_df['rides_per_day'] = grouped_df['total_rides'] / grouped_df['days_in_operation']
grouped_df['revenue_per_day'] = grouped_df['rides_per_day'] * grouped_df['avg_ride_price']

# Dropping a couple stations with very few rides 
grouped_df = grouped_df[grouped_df['total_rides'] >= 100]



################## Some visualizations ##################


# Scatter plot 1 

plt.scatter(grouped_df['revenue_per_day'], grouped_df['electric_bike_percentage'], s=50, alpha=0.7)

# Add labels and title to the plot
plt.xlabel('Average Revenue Per Day')
plt.ylabel('Electric Bike Percentage')
plt.title('Scatter Plot of Ride Time vs. Electric Bike Percentage')
plt.show()

# Scatter plot 2 

grouped_df['log_avg_ride_price'] = np.log(grouped_df['avg_ride_price'])

# Scatter plot with the log-transformed average ride price
plt.scatter(grouped_df['log_avg_ride_price'], grouped_df['electric_bike_percentage'], s=50, alpha=0.7)

# Add labels and title to the plot
plt.xlabel('Log Average Ride Price')
plt.ylabel('Electric Bike Percentage')
plt.title('Scatter Plot of Log Ride Price vs. Electric Bike Percentage')

# Calculate the best-fit line using polyfit
coefficients = np.polyfit(grouped_df['log_avg_ride_price'], grouped_df['electric_bike_percentage'], 1)
best_fit_line = np.polyval(coefficients, grouped_df['log_avg_ride_price'])

# Plot the best-fit line
plt.plot(grouped_df['log_avg_ride_price'], best_fit_line, color='red', linewidth=2)

plt.show()




# Histogram 1 

# Create the histogram
plt.hist(grouped_df['electric_bike_percentage'], bins=20, edgecolor='black')

# Set the labels and title
plt.xlabel('Electric Bike Percentage')
plt.ylabel('Frequency')
plt.title('Histogram of Electric Bike Percentage')

# Display the plot
plt.show()


# Histogram 2 


# Create the histogram
plt.hist(grouped_df['log_avg_ride_price'], bins=20, edgecolor='black')

# Set the labels and title
plt.xlabel('Log of Average ride price')
plt.ylabel('Frequency')
plt.title('Histogram of Log of Average Ride Price')

# Display the plot
plt.show()




