Depends on AstroPy, NumPy, and MatPlotLib

## Functions

### def filesFromDir(keywords, path=None):
Purpose:    extract file names containing keyword strings from a directory

Parameters: list of keyword strings KEYWORDS; 
            optional directory path string PATH;
            if no PATH given, searches the current directory
            
Returns:    list of file name strings matching keywords in directory

### def mergeTables(file_list, names, selected):
Purpose:    read list of input files to single data table
Parameters: list of file names to read FILE_LIST;
            list of table column name strings NAMES;
            string name of selected column SELECTED
Returns:    NumPy table of input data
            Columns represent input files
            Rows represent time slices

### def jacknifeBinData(data):
Purpose:    calculate table of jacknife bins for input data
Parameters: NumPy table DATA:
            Columns represent trials
            Rows represent time slices
Returns:    NumPy table
            Columns represent jacknife bins
            Rows represent time slices

### def effectiveEnergy(bins):
Purpose:    calculate effective energy from 2-point function values
Parameters: NumPy table BINS:
            Columns represent jacknife bin values of 2-point function
            Rows represent time slices
Returns:    NumPy table
            Columns represent jacknife bin values for effective energy
            Rows represent time slices

### def averageColumns(vals):
Purpose:    average values over table columns
Parameters: NumPy table VALS:
            Columns of values to be averaged
            Rows represent time slices
Returns:    1D NumPy array
            Lists mean values of each input row
