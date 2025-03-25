# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:42:57 2025

@author: MahdaviAl
"""


import pandas as pd
from hobo_read_in import ux_100_001_read
import os
import glob
import matplotlib.pyplot as plt



main_folder = r'C:\logger_data\curing_room'
os.chdir(main_folder)
csv_files = glob.glob('*.csv', recursive=True)
# csv_files.remove('4013_test_F.csv')
output_folder_main = r'C:\logger_data\processed\trh\\'

# Read all sensor data and append
df_all = pd.DataFrame([])
for file in csv_files:
    output_file = os.path.join(output_folder_main, file[:-4] + '_processed.xlsx')
    temp = ux_100_001_read(file, output_path = output_file, format_ = 'csv', t_scale = 'C', save = True)
    temp['Sensor'] = file[:-4]
    df_all = pd.concat([df_all, temp], axis = 0)


# Time-stamp date/time
df_all['Time'] = pd.to_datetime(df_all['Time'])
df_all['Sensor'] = df_all['Sensor'].str.split('_').str[0]


# Trim time
start_time = pd.to_datetime('2025-02-05 15:05:00')
end_time = pd.to_datetime('2025-02-05 16:20:00')
df_all = df_all[(df_all['Time'] >= start_time) & (df_all['Time'] <= end_time)]

dict_color = dict(zip(df_all['Sensor'].unique(), ['red', 'blue', 'green', 'orange', 'purple', 
                                                  'brown', 'pink', 'cyan', 'navy', 
                                                  'maroon', 'deeppink', 'violet', 'indigo',
                                                  'khaki', 'gold', 'aquamarine']))

# Plotting multiple series
plt.figure(figsize = (12, 6))


# Plot each column except the Date
for s in df_all['Sensor'].unique():
    temp2 = df_all[df_all['Sensor'] == s]
    plt.plot(temp2['Time'], temp2['Temp'], label = s, color = dict_color[s])

# Customize the plot
plt.xlabel('Time (min)')
plt.ylabel('Temperature (°C)')
plt.legend()

plt.tight_layout()

# Display the plot
plt.savefig(r'C:\logger_data\processed\trh_sensor.jpg', dpi = 1200, bbox_inches = 'tight')
plt.show()


# Plotting multiple series
plt.figure(figsize = (12, 6))


# Plot each column except the Date
for s in df_all['Sensor'].unique():
    temp2 = df_all[df_all['Sensor'] == s]
    plt.plot(temp2['Time'], temp2['RH'], label = s, color = dict_color[s])

# Customize the plot
plt.xlabel('Time (min)')
plt.ylabel('Temperature (°C)')
plt.legend()

plt.tight_layout()

# Display the plot
plt.savefig(r'C:\logger_data\processed\rh_sensor.jpg', dpi = 1200, bbox_inches = 'tight')
plt.show()



