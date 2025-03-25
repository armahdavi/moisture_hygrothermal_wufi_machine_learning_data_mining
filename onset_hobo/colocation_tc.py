# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 12:43:10 2025

@author: MahdaviAl
"""

import pandas as pd
from hobo_read_in import ux_100_03m_read, ux_100_001_read
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

###################################
### Step 1: Essential functions ###
###################################

def in_folder_hobo_combine(in_folder, out_folder, start_time, end_time, sensor_type = 'tc'):
    '''
    Description
    -----------
    Combines and processes sensor data from CSV files in a given output folder.

    Parameters
    ----------
    in_folder : str
        Path to the input folder (currently unused in the function).
    out_folder : str
        Path to the output folder where CSV files are located and processed Excel files will be saved.
    start_time : str
        Start time for filtering or processing data (currently not used).
    end_time : str
        End time for filtering or processing data (currently not used).

    Returns
    -------
    df_all : pandas.DataFrame
        A DataFrame containing the combined data from all processed CSV files.

    '''
    
    # Set directory and list csv files 
    os.chdir(in_folder)
    csv_files = glob.glob('*.csv', recursive=True)

    # Read all sensor data and append
    df_all = pd.DataFrame([])
    for file in csv_files:
        os.makedirs(out_folder, exist_ok = True)
        output_file = os.path.join(out_folder, file[:-4] + '_processed.xlsx')
        if sensor_type == 'tc':
            df = ux_100_03m_read(file, output_path = output_file, format_ = 'csv', t_scale = 'C', save = True)
        elif sensor_type == 'trh':
            df = ux_100_001_read(file, output_path = output_file, format_ = 'csv', t_scale = 'C', save = True)
        else:
            raise ValueError("Sensor type must be 'tc' or 'trh'.")
        df['file'] = file[:-4]
        df_all = pd.concat([df_all, df], axis = 0)

    # Time-stamp time and trim
    df_all['Time'] = pd.to_datetime(df_all['Time'])
    df_all = df_all[(df_all['Time'] >= start_time) & (df_all['Time'] <= end_time)]

    return df_all


def plot_sensor_colocate_t(df, out_folder, fig_name, format_ = 'jpg', sensor_type = 'tc'):
    '''
    Description
    -----------
    Plots sensor temperature data over time and saves the figure to a specified output folder.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the sensor data. Must include columns 'Sensor', 'Time', and 'T-Type'.
    out_folder : str
        Path to the output folder where the plot will be saved.
    fig_name : str
        Name of the output figure file (without extension).
    format_ : str, optional
        File format for saving the figure (e.g., 'jpg', 'png'). Default is 'jpg'.

    Returns
    -------
    None. Only sketches the graph
    '''
    
    if not 'Sensor' in df.columns:
        raise ValueError("The DataFrame must contain a 'Sensor' column.")
    else:
        # Create random colors in the number of available sensor data 
        dict_color = dict(zip(df['Sensor'].unique(), sns.color_palette('hsv', len(df['Sensor'].unique()))))
        
        plt.figure(figsize = (12, 6))
    
        # Plot each column in df except the Date
        for s in df['Sensor'].unique():
            temp = df[df['Sensor'] == s]
            
            if sensor_type == 'tc':
                plt.plot(temp['Time'], temp['T-Type'], label = s, color = dict_color[s])
            elif sensor_type == 'trh':
                plt.plot(temp['Time'], temp['Temp'], label = s, color = dict_color[s])
                # plt.plot(temp['Time'], temp['RH'], label = s, color = dict_color[s])
            else:
                raise ValueError("Sensor type must be 'tc' or 'trh'.")
                
        # Customize the plot
        plt.xlabel('Time (min)')
        plt.ylabel('Temperature (°C)')
        plt.legend()
    
        plt.tight_layout()
        
        output_file = os.path.join(out_folder, fig_name + f'.{format_}')
        os.makedirs(os.path.join(out_folder), exist_ok = True)    
        # Display the plot
        plt.savefig(output_file, dpi = 1200, bbox_inches = 'tight')
        plt.show()

   
        
def plot_sensor_colocate_rh(df, out_folder, fig_name, format_ = 'jpg', sensor_type = 'tc'):
    '''
    Description
    -----------
    Plots sensor RH data over time and saves the figure to a specified output folder.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the sensor data. Must include columns 'Sensor', 'Time', and 'RH'.
    out_folder : str
        Path to the output folder where the plot will be saved.
    fig_name : str
        Name of the output figure file (without extension).
    format_ : str, optional
        File format for saving the figure (e.g., 'jpg', 'png'). Default is 'jpg'.

    Returns
    -------
    None. Only sketches the graph
    '''
    
    if not 'Sensor' in df.columns:
        raise ValueError("The DataFrame must contain a 'Sensor' column.")
    else:
        # Create random colors in the number of available sensor data 
        dict_color = dict(zip(df['Sensor'].unique(), sns.color_palette('hsv', len(df['Sensor'].unique()))))
        
        plt.figure(figsize = (12, 6))
    
        # Plot each column in df except the Date
        for s in df['Sensor'].unique():
            temp = df[df['Sensor'] == s]
            
            if sensor_type != 'trh':
                raise ValueError("Sensor type must be 'trh'.")
            else:
                plt.plot(temp['Time'], temp['RH'], label = s, color = dict_color[s])
                           
                
        # Customize the plot
        plt.xlabel('Time (min)')
        plt.ylabel('Relative Humidity (RH) (%)')
        plt.legend()
    
        plt.tight_layout()
        
        output_file = os.path.join(out_folder, fig_name + f'.{format_}')
        os.makedirs(os.path.join(out_folder), exist_ok = True)    
        # Display the plot
        plt.savefig(output_file, dpi = 1200, bbox_inches = 'tight')
        plt.show()



#########################
### Step 2: Use cases ###
#########################

# Use case 1: First tc sensor colocation 
in_folder = r'C:\EXP\logger_data\raw\tc'
out_folder = r'C:\EXP\logger_data\processed\first_tc_colocation_250205'
start_time = pd.to_datetime('2025-02-04 18:00:00')
end_time = pd.to_datetime('2025-02-05 16:00:00')

df = in_folder_hobo_combine(in_folder, out_folder, start_time, end_time)
df = df[~(df['file'] == '4013_test_F')]
df['Sensor'] = df['file'].str.split('_').str[0]

## Sketch colocation plot
out_folder = r'C:\EXP\logger_data\processed'
fig_name = 'first_tc_colocation_250205'
plot_sensor_colocate_t(df, out_folder, fig_name)


# Use case 2: First trh sensor colocation (UX-100-001)
in_folder = r'C:\EXP\logger_data\raw\trh_ux'
out_folder = r'C:\EXP\logger_data\processed\first_trh_colocation_250205'
start_time = pd.to_datetime('2025-02-04 18:30:00')
end_time = pd.to_datetime('2025-02-05 10:00:00')

df = in_folder_hobo_combine(in_folder, out_folder, start_time, end_time, sensor_type = 'trh')
df['Sensor'] = df['file'].str.split('_').str[0]

## Sketch colocation plot
out_folder = r'C:\EXP\logger_data\processed'
fig_name = 'first_trh_colocation_250205'
plot_sensor_colocate_t(df, out_folder, fig_name, sensor_type = 'trh')


# Use case 3: one u12 and two ux trh sensors
in_folder = r'C:\EXP\logger_data\raw\trh_u12_ux'
out_folder = r'C:\EXP\logger_data\processed\trh_colocation_ux_u12_250205'
start_time = pd.to_datetime('2025-02-05 11:30:00')
end_time = pd.to_datetime('2025-02-05 15:45:00')

df = in_folder_hobo_combine(in_folder, out_folder, start_time, end_time, sensor_type = 'trh')
df['Sensor'] = df['file'].str.split('_').str[0]

## Sketch colocation plot (temp)
out_folder = r'C:\EXP\logger_data\processed'
fig_name = 'trh_ux_u12_colocation_t_250205'
plot_sensor_colocate_t(df, out_folder, fig_name, sensor_type = 'trh')

## Sketch colocation plot (rh)
fig_name = 'trh_ux_u12_colocation_RH_250205'
plot_sensor_colocate_rh(df, out_folder, fig_name, sensor_type = 'trh')


# Use case 4: Colocation at home
in_folder = r'C:\EXP\logger_data\raw\home'
out_folder = r'C:\EXP\logger_data\processed\home'
start_time = pd.to_datetime('2025-02-05 21:30:00')
end_time = pd.to_datetime('2025-02-06 08:30:00')

df = in_folder_hobo_combine(in_folder, out_folder, start_time, end_time, sensor_type = 'trh')
df['Sensor'] = df['file'].str.split('_').str[0]

## Sketch colocation plot (Temp)
out_folder = r'C:\EXP\logger_data\processed'
fig_name = 'home_colocation_t'
plot_sensor_colocate_t(df, out_folder, fig_name, sensor_type = 'trh')

## Sketch colocation plot (RH)
fig_name = 'home_colocation_rh'
plot_sensor_colocate_rh(df, out_folder, fig_name, sensor_type = 'trh')


# Use case 5: Curing room
in_folder = r'C:\EXP\logger_data\raw\curing_room'
out_folder = r'C:\EXP\logger_data\processed\curing_room'
start_time = pd.to_datetime('2025-02-05 15:00:00')
end_time = pd.to_datetime('2025-02-05 16:20:00')

df = in_folder_hobo_combine(in_folder, out_folder, start_time, end_time, sensor_type = 'trh')
df['Sensor'] = df['file'].str.split('_').str[0]

## Sketch colocation plot (temp)
out_folder = r'C:\EXP\logger_data\processed'
fig_name = 'curing_room_ux_t'
plot_sensor_colocate_t(df, out_folder, fig_name, sensor_type = 'trh')

## Sketch colocation plot (RH)
out_folder = r'C:\EXP\logger_data\processed'
fig_name = 'curing_room_ux_rh'
plot_sensor_colocate_rh(df, out_folder, fig_name, sensor_type = 'trh')



# use case 6: U12 colocations
## Note: 2 ux are also there for 
in_folder = r'C:\EXP\logger_data\raw\trh_u12'
out_folder = r'C:\EXP\logger_data\processed\second_trh_colocation_u12'
start_time = pd.to_datetime('2025-02-06 13:15:00')
end_time = pd.to_datetime('2025-02-06 16:45:00')

df = in_folder_hobo_combine(in_folder, out_folder, start_time, end_time, sensor_type = 'trh')
df['Sensor'] = df['file'].str.split('_').str[0]

## Sketch colocation plot (temp)
out_folder = r'C:\EXP\logger_data\processed'
fig_name = 'trh_u12_colocation_t'
plot_sensor_colocate_t(df, out_folder, fig_name, sensor_type = 'trh')

## Sketch colocation plot (temp)
out_folder = r'C:\EXP\logger_data\processed'
fig_name = 'trh_u12_colocation_rh'
plot_sensor_colocate_rh(df, out_folder, fig_name, sensor_type = 'trh')


# use case 7: New TC wire sensors
in_folder = r'C:\EXP\logger_data\raw\new_tc_wire'
out_folder = r'C:\EXP\logger_data\processed\new_tc_colocation_250210'
start_time = pd.to_datetime('2025-02-10 12:00:00')
end_time = pd.to_datetime('2025-02-10 16:00:00')

df = in_folder_hobo_combine(in_folder, out_folder, start_time, end_time)
# df = df[~(df['file'] == '4013_test_F')]
df['Sensor'] = df['file'].str.split('_').str[0]

## Sketch colocation plot
out_folder = r'C:\EXP\logger_data\processed'
fig_name = 'new_tc_colocation_250210'
plot_sensor_colocate_t(df, out_folder, fig_name)
