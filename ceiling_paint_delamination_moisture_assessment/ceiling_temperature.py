# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 14:02:28 2025

@author: MahdaviAl
"""

import pandas as pd
import os
import matplotlib.pyplot as plt

########################################################
### Step 1: Data ingestion and variable deifinitions ###
########################################################

folder = r'temperature_data'
save_folder = r'temperature_data\processed'

df_pre = pd.read_excel(os.path.join(folder, 'site_DES.xlsx'), sheet_name = 'IR')
df_post = pd.read_excel(os.path.join(folder, 'site_DES.xlsx'), sheet_name = 'IR_0228')

# Assign Suite No. to a color for plotting
dict_col = {'104': 'r',
            '108': 'b',
            '204': 'g',
            '312': 'yellow',
            '405': 'purple',
            '414': 'brown',
            '514': 'pink',
            '608': 'Lightblue',
            '701': 'grey',
            '812': 'blueviolet'}

######################################
### Step 2: Plotting Visualization ###
######################################

# For loop for pre and post deployment ceiling temperature analysis
for df in [df_pre, df_post]:
    del df['Comment']
    df = df.pivot(index = 'Suite', columns = 'Loc', values = 'temp')
    df = df.T
    df.columns = [str(col) for col in df.columns]

    # Plot T vs. suite
    plt.figure(figsize = (8,4))
    for col in df.columns:
        plt.plot(df.index, df[col], 
                 marker = 'o', 
                 label = col, 
                 c = dict_col[col],
                 markeredgecolor = dict_col[col],
                 markerfacecolor = 'none', 
                 ls = '--')
    
    # Axes properties
    plt.xlabel('Ceiling Location', fontsize = 14)
    plt.ylabel('Temperature (°C)', fontsize = 14)
    plt.ylim(16, 32)
    
    
    # Scatter plot added representing average T
    df['Average'] = df.mean(axis = 1)
    plt.scatter(df.index, df['Average'], marker = 'x', label = 'Average', c = 'k', s = 200)
    
    # Other non-common properties and saving
    if df == df_pre:
        plt.title('Ceiling Temperature (Initial)', fontsize = 14, weight = 'bold')
        plt.legend(bbox_to_anchor = (1.2, 0.8), ncol = 1)
        plt.savefig(os.path.join(save_folder, 'pre_deploy_ceiling.jpg'), bbox_inches='tight', pad_inches = 0.1, dpi = 1200)
    else:
        plt.title('Ceiling Temperature (Final)', fontsize = 14, weight = 'bold')
        plt.legend(bbox_to_anchor = (1.2, 0.9), ncol = 1)
        plt.savefig(os.path.join(save_folder, 'post_deploy_ceiling.jpg'), bbox_inches='tight', pad_inches = 0.1, dpi = 1200)
        
    plt.show()