# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 10:55:05 2025

@author: MahdaviAl
"""

import os
os.chdir(r'C:\python_projects\ieq')
import pandas as pd
from hobo_read_in import ux_100_001_read, ux_100_03m_read
import glob
import re
import psychrolib
import matplotlib.pyplot as plt
import numpy as np


##################################################
### Step 1: Functions and Variable Initiations ###
##################################################

psychrolib.SetUnitSystem(psychrolib.SI)

def calculate_RH2(row, rh1_col, t1_col, t2_col, pressure=101325): #added pressure
    RH1 = row[rh1_col]
    T1 = row[t1_col]
    T2 = row[t2_col]

    # Calculate the humidity ratio at T1 and RH1
    W1 = psychrolib.GetHumRatioFromRelHum(T1, RH1/100, pressure) #RH1 should be a fraction, not a percent.

    # Calculate the new relative humidity at T2 with the same humidity ratio
    RH2 = psychrolib.GetRelHumFromHumRatio(T2, W1, pressure)

    return RH2*100 #return the RH2 in percentage.

mother_folder = r'temperature_data'
trh = os.path.join(mother_folder, 'raw/u12_trh')
tcp = os.path.join(mother_folder, 'raw/ux_tc')

start_date = '2025-02-14 13:00:00'
end_date = '2025-02-28 13:00:00'


########################################
### Step 2: Processing T and RH Data ### 
########################################

# Bulk air T and RH
df_all1 = pd.DataFrame([])

for file in glob.glob(os.path.join(trh, "*.xlsx")):
    df = ux_100_001_read(file, output_path = None, format_ = 'xlsx', t_scale = 'C', save = False)
    match = re.search(r'_(.*?)\.', os.path.basename(file))
    df['Suite'] = match.group(1)
    
    df_all1 = pd.concat([df_all1, df], axis = 0)

df_all1 = df_all1[(df_all1['Time'] >= start_date) & (df_all1['Time'] <= end_date)]

# Ceiling surface T and RH
df_all2 = pd.DataFrame([])
for file in glob.glob(os.path.join(tcp, "*.xlsx")):
    df = ux_100_03m_read(file, output_path = None, format_ = 'xlsx', t_scale = 'C', save = False)
    match = re.search(r'_(.*?)\.', os.path.basename(file))
    df['Suite'] = match.group(1)
    
    df_all2 = pd.concat([df_all2, df], axis = 0)

df_all2 = df_all2[(df_all2['Time'] >= start_date) & (df_all2['Time'] <= end_date)]
df_all2['T-Type'] = df_all2['T-Type'].combine_first(df_all2['K-Type'])
df_all2.drop(['Temp', 'K-Type'], axis = 1, inplace = True)


# Merge bulk and surface into one df
df_all = pd.merge(df_all1, df_all2, on = ['Time', 'Suite'], how = 'inner')
df_all['RH_surface'] = df_all.apply(calculate_RH2, axis=1,  args=('RH', 'Temp', 'T-Type'))
df_all['Diff'] = df_all['Temp'] -  df_all['T-Type']
df_all.sort_values(['Suite', 'Time'], inplace = True)

output_file = os.path.join(mother_folder, 'processed', 'trh_data.xlsx')
df_all.to_excel(output_file, index = False)


###################################
### Step 3: Data Visualizations ### 
###################################

# 3A: PLOT TEMPERATURE DATA
plt.figure(figsize=(8, 6))

## Positions of boxes
positions_A = np.arange(len(df_all['Suite'].unique())) * 2
positions_B = positions_A + 1

## Boxes of Bulk T
box_A = plt.boxplot(
    [df_all.loc[df_all['Suite'] == cat, 'Temp'] 
     for cat in df_all['Suite'].unique()],
    positions=positions_A, widths=0.4, patch_artist=True,
    boxprops=dict(facecolor='skyblue', color='k'),
    medianprops=dict(color='k'),
    flierprops=dict(marker='x', markersize=2, markerfacecolor='k')  # Smaller outlier markers

)

## Boxes of Surface T
box_B = plt.boxplot(
    [df_all.loc[df_all['Suite'] == cat, 'T-Type'] 
     for cat in df_all['Suite'].unique()],
    positions=positions_B, widths=0.4, patch_artist=True,
    boxprops=dict(facecolor='orange', color='k'),
    medianprops=dict(color='k'),
    flierprops=dict(marker='x', markersize=2, markerfacecolor='k')  # Smaller outlier markers

)

## x and y axis properties 
midpoints = (positions_A + positions_B) / 2
plt.xticks(midpoints, df_all['Suite'].unique(), fontsize=10)
plt.xlabel('Suite No')

plt.ylabel('Temperature (°C)')

## Other graph properties
for x_val in positions_B:
    plt.axvline(x=x_val+0.5, color='k', linestyle='--', linewidth=0.25)

plt.legend([box_A['boxes'][0], box_B['boxes'][0]], 
           ['Bulk T', ' Ceiling Surface T'], 
           loc='upper right',
           frameon=True, edgecolor='black')

plt.tight_layout()
plt.savefig(os.path.join(mother_folder, 'processed', 'temperature.jpg'), 
            bbox_inches='tight', pad_inches = 0.1, dpi = 1200)

plt.show()




# 3B: PLOT RELATIVE HUMIDITY DATA
plt.figure(figsize=(8, 6))

## Boxes of Bulk RH
box_A = plt.boxplot(
    [df_all.loc[df_all['Suite'] == cat, 'RH'] 
     for cat in df_all['Suite'].unique()],
    positions=positions_A, widths=0.4, patch_artist=True,
    boxprops=dict(facecolor='skyblue', color='k'),
    medianprops=dict(color='k'),
    flierprops=dict(marker='x', markersize=2, markerfacecolor='k')  # Smaller outlier markers

)

## Boxes of Surface RH
box_B = plt.boxplot(
    [df_all.loc[df_all['Suite'] == cat, 'RH_surface'] 
     for cat in df_all['Suite'].unique()],
    positions=positions_B, widths=0.4, patch_artist=True,
    boxprops=dict(facecolor='orange', color='k'),
    medianprops=dict(color='k'),
    flierprops=dict(marker='x', markersize=2, markerfacecolor='k')  # Smaller outlier markers

)

## x and y axis properties 
midpoints = (positions_A + positions_B) / 2
plt.xticks(midpoints, df_all['Suite'].unique(), fontsize=10)
plt.xlabel('Suite No')

plt.ylabel('Relative Humidity (RH) (%)')
plt.ylim(0, 100)
plt.yticks(range(0, 101, 20))


## Other graph properties
for x_val in positions_B:
    plt.axvline(x=x_val+0.5, color='k', linestyle='--', linewidth=0.25)

plt.legend([box_A['boxes'][0], box_B['boxes'][0]], 
           ['Bulk RH', 'Ceiling Surface RH'], 
           loc='upper right',
           frameon=True, edgecolor='black')

plt.axhspan(70, 100, facecolor='red', alpha=0.15)  # Highlight area
plt.text(0.5,85, 'Critical RH for Paint Delimination, > 70%', color = 'r')
plt.tight_layout()

plt.savefig(os.path.join(mother_folder, 'processed', 'relative_humidity.jpg'), 
            bbox_inches='tight', pad_inches = 0.1, dpi = 1200)

plt.show()

