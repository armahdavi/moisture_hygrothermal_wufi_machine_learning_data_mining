# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:43:36 2025

@author: MahdaviAl
"""

import pandas as pd
import re


def ux_100_03m_read(input_path, output_path = None, format_ = 'csv', t_scale = 'C', save = True):
    '''
    Parameters
    ----------
    input_path : string path of input file (file name included)
    output_path : string path of output file (file name included, preprocessing needed)
    format_ : input file format (only csv or xls/xlsx is accepted)       
    t_scale : C or F
    save: whether saving the output is requested
       

    Returns
    -------
    dataframe of cleaned time-series sensor data

    '''
        
    # If save is True, ensure output_path is provided
    if save and not output_path:
        raise ValueError('Output path is required when save = True.')
    
    # Check file formatting
    if format_ == 'csv':
        df = pd.read_csv(input_path, skiprows = 1)
    elif (format_ == 'xlsx') | (format_ == 'xls'):
        df = pd.read_excel(input_path, skiprows = 1)
    else:
        raise ValueError('Not a common HOBO raw file.')
    
    # Clean up logg rows
    df = df[~df.apply(lambda row: row.astype(str).str.contains('logged', case = False, na = False).any(), axis = 1)]
        
    # Correct temperature based on function input
    col_list_correction = [col for col in df.columns if ('°C' in col) | ('°F' in col)]
    
    if t_scale == 'C':
        for col in col_list_correction:
            if not '°C' in col:
                df[col] = (df[col] - 32)/1.8
    
    elif t_scale == 'F': 
        for col in col_list_correction:
            if not '°F' in col:
                df[col] = 1.8 * df[col] + 32
    else:
        raise ValueError('Not a common temperature unit.')
    
    # Keep necessary columns
    keep_list = [df.columns[1]] + col_list_correction
    df = df[keep_list]
    col_list_correction = [re.split(r', °[CF]', s)[0] for s in col_list_correction]
    df.columns = ['Time'] + col_list_correction

    
    if save:
        df.to_excel(output_path, index = False)
    
    return df
    

# Test fucntion
input_path = r'C:\EXP\logger_data\raw\tc\4013_test.csv'
df = ux_100_03m_read(input_path, save = False)


    
def ux_100_001_read(input_path, output_path = None, format_ = 'csv', t_scale = 'C', save = True):
    '''
    Parameters (CORRECT)
    ----------
    input_path : string path of input file (file name included)
    output_path : string path of output file (file name included, preprocessing needed)
    format_ : input file format (only csv or xls/xlsx is accepted)       
    t_scale : C or F
    save: whether saving the output is requested
       

    Returns
    -------
    dataframe of cleaned time-series sensor data

    '''
        
    # If save is True, ensure output_path is provided
    if save and not output_path:
        raise ValueError('Output path is required when save = True.')
    
    # Check file formatting
    if format_ == 'csv':
        df = pd.read_csv(input_path, skiprows = 1)
    elif (format_ == 'xlsx') | (format_ == 'xls'):
        df = pd.read_excel(input_path, skiprows = 1)
    else:
        raise ValueError('Not a common HOBO raw file.')
    
    # Clean up logg rows
    df = df[~df.apply(lambda row: row.astype(str).str.contains('logged', case = False, na = False).any(), axis = 1)]
        
    # Correct temperature based on function input
    col_list_correction = [col for col in df.columns if ('°C' in col) | ('°F' in col) | ('RH, %' in col)]
    
    if t_scale == 'C':
        col_list_correction2 = [col for col in df.columns if ('°C' in col) | ('°F' in col)]
        for col in col_list_correction2:
            if not '°C' in col:
                df[col] = (df[col] - 32)/1.8
    
    elif t_scale == 'F': 
        for col in col_list_correction2:
            if not '°F' in col:
                df[col] = 1.8 * df[col] + 32
    else:
        raise ValueError('Not a common temperature unit.')
    
    # Keep necessary columns
    keep_list = [df.columns[1]] + col_list_correction
    df = df[keep_list]
    col_list_correction = [re.split(r', (°[CF]|%)', s)[0] for s in col_list_correction]
    df.columns = ['Time'] + col_list_correction

    
    if save:
        df.to_excel(output_path, index = False)
    
    return df
    
    
def mx_1101_read(input_path, output_path = None, format_ = 'csv', t_scale = 'C', save = True):
    '''
    Parameters (CORRECT)
    ----------
    input_path : string path of input file (file name included)
    output_path : string path of output file (file name included, preprocessing needed)
    format_ : input file format (only csv or xls/xlsx is accepted)       
    t_scale : C or F
    save: whether saving the output is requested
       

    Returns
    -------
    dataframe of cleaned time-series sensor data

    '''
        
    # If save is True, ensure output_path is provided
    if save and not output_path:
        raise ValueError('Output path is required when save = True.')
    
    # Check file formatting
    if format_ == 'csv':
        df = pd.read_csv(input_path, skiprows = 1)
    elif (format_ == 'xlsx') | (format_ == 'xls'):
        df = pd.read_excel(input_path, skiprows = 1)
    else:
        raise ValueError('Not a common HOBO raw file.')
    
    # Clean up logg rows
    df = df[~df.apply(lambda row: row.astype(str).str.contains('logged', case = False, na = False).any(), axis = 1)]
        
    # Correct temperature based on function input
    col_list_correction = [col for col in df.columns if ('°C' in col) | ('°F' in col) | ('RH, %' in col)]
    
    if t_scale == 'C':
        col_list_correction2 = [col for col in df.columns if ('°C' in col) | ('°F' in col)]
        for col in col_list_correction2:
            if not '°C' in col:
                df[col] = (df[col] - 32)/1.8
    
    elif t_scale == 'F': 
        for col in col_list_correction2:
            if not '°F' in col:
                df[col] = 1.8 * df[col] + 32
    else:
        raise ValueError('Not a common temperature unit.')
    
    # Keep necessary columns
    keep_list = [df.columns[1]] + col_list_correction
    df = df[keep_list]
    col_list_correction = [re.split(r', (°[CF]|%)', s)[0] for s in col_list_correction]
    df.columns = ['Time'] + col_list_correction

    
    if save:
        df.to_excel(output_path, index = False)
    
    return df
    
        
    