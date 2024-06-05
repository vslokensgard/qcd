from astropy.io import ascii
from astropy import table
from astropy.table import Table
from astropy.table import Column
import numpy as np
import matplotlib.pyplot as plt
import os

# Purpose:    extract file names containing keyword strings from a directory
# Parameters: list of keyword strings KEYWORDS
#             optional directory path string PATH
#             if no PATH given, searches the current directory
# Returns:    list of file name strings matching keywords in directory
def filesFromDir(keywords, path=None):
    if path:
        all_files = os.listdir(path)
    else:
        all_files = os.listdir()
    
    subset = [k for k in all_files if any(keyword in k for keyword in keywords)]
    return subset

# Purpose:    read list of input files to single data table
# Parameters: list of file names to read FILE_LIST;
#             list of table column name strings NAMES;
#             string name of selected column SELECTED
# Returns:    NumPy table of input data
#             Columns represent input files
#             Rows represent time slices
def mergeTables(file_list, names, selected):
    merged = []
    
    # Add selected columns from each file to a single table 
    for i in range(len(file_list)):
        current_file = ascii.read(file_list[i])
        current_table = Table(data=current_file, names=names)
        trial_data = Column(data=current_table[selected], name=str(i)).data
        merged.append(trial_data)

    merged_array = np.asarray(merged)
    final_array = np.transpose(merged_array)
    return final_array

# Purpose:    calculate table of jacknife bins for input data
# Parameters: NumPy table DATA:
#             Columns represent trials
#             Rows represent time slices
# Returns:    NumPy table
#             Columns represent jacknife bins
#             Rows represent time slices
def jacknifeBinData(data):
    binned = np.copy(data)
    num_slices, num_bins = data.shape
    
    for i in range(num_slices):
        row_sum = np.sum(data[i])
        
        for j in range(num_bins):
            binned[i][j] = row_sum - data[i][j] / num_bins
      
    return binned

# Purpose:    calculate effective energy from 2-point function values
# Parameters: NumPy table BINS:
#             Columns represent jacknife bin values of 2-point function
#             Rows represent time slices
# Returns:    NumPy table
#             Columns represent jacknife bin values for effective energy
#             Rows represent time slices
def effectiveEnergy(bins):
    num_intervals = bins.shape[0] - 1
    num_bins = bins.shape[1]
    in_by_bin = np.transpose(bins)
    out_by_bin = np.zeros((num_bins, num_intervals))
    
    if (np.any(in_by_bin <= 0)):
        print("Warning: binned data includes negative or zero values.")
        print("Effective energies for these intervals will be set to 0.")
    
    for bin_no in range(num_bins):
        
        for int_no in range(num_intervals):
            if (in_by_bin[bin_no][int_no] <= 0) or (in_by_bin[bin_no][int_no+1] <= 0):
                out_by_bin[bin_no][int_no] = 0
            else:
                out_by_bin[bin_no][int_no] = np.log(in_by_bin[bin_no][int_no] / in_by_bin[bin_no][int_no+1])
    
    final_bins = np.transpose(out_by_bin)
    return final_bins

# Purpose:    average values over table columns
# Parameters: NumPy table VALS:
#             Columns of values to be averaged
#             Rows represent time slices
# Returns:    1D NumPy array
#             Lists mean values of each input row
def averageColumns(vals):
    num_intervals, num_trials = vals.shape
    avg = np.sum(vals, axis=1)
    avg /= num_trials
    
    return avg
