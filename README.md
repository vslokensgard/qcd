# pionkaon.py

Depends on NumPy & MatPlotLib

## Functions

### def filesFromDir(keywords, path=None):
Purpose:    extract file names containing keyword strings from a directory
Parameters: list of keyword strings KEYWORDS
            optional directory path string PATH (must end in a "/")
            if no PATH given, searches the current directory
Returns:    list of file name strings matching keywords in directory

### def mergeTables(file_list, names, selected):
Purpose:    read list of input files to single data table
Parameters: list of file names to read FILE_LIST;
            index of selected column COL_INDEX
Returns:    NumPy table of input data
            Columns represent input files
            Rows represent time slices

### def jacknifeAverage(data):
Purpose:    calculate table of jacknife bins for input data
Parameters: NumPy table DATA:
            Columns represent trials
            Rows represent time slices
Returns:    NumPy table:
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

### def jacknifeError2D(binned_data, rho=None):
Purpose:    calculate jacknife error across all bins for each timeslice
Parameters: NumPy table BINNED_DATA:
            rows representing timeslices,
            columns representing bins
Returns:    NumPy 1D array:
            one error value per timeslice

### def maxSlice(jacknife_err, max_fact):
Purpose:    identify the first timeslice whose error exceeds a given threshold
Parameters: 1D Numpy array of error values JACKNIFE_ERR
            number MAX_FACT, product with the error of index 1 of JACKNIFE_ERR will set the threshold
Returns:    index of first value in JACKNIFE_ERR greater than the threshold, or 5, whichever is larger

### def jacknifeGraph(eff_energies, jack_error, err_cutoff=None, selected_bins=None):
Purpose:    graph effective energy values with jacknife error for each bin
Parameters: 2D NumPy array EFF_ENERGIES:
            rows represent timeslices,
            columns represent bins;
            1D NumPy array JACK_ERROR:
            represents error values associated with each bin value
            number ERR_CUTOFF representing max error as a factor of the error of JACK_ERROR[1]
            list of integers SELECTED_BINS, if not None then only graph listed indeces
Returns:    void

### def minSlice(ycoords, yerr, length=3):
Purpose:    identify the start indexes of plateaus of a given length, excluding index 0
Parameters: 1D NumPy array representing the bin values for each timeslice YCOORDS
            1D NumPy array representing jacknife errors for each timeslice YERR
            optional number LENGTH representing minimum length of the plateau
Returns:    start index of the plateau, 0 if no plateau is found

### def listPlateaus(binned_vals, binned_err, length=3, selected_bins=None):
Purpose:    wrapper function to run multiple bins of data through minSlice()
Parameters: 2D NumPy array of timeslice rows & bin columns BINNED_VALS
            1D NumPy array representing jacknife errors for each timeslice BINNED_ERR
            optional number LENGTH representing minimum length of the plateau
            optional list of bin indexes SELECTED_BINS to test, if None then test all
Returns:    1D NumPy array of the plateau start index for each selected bin

### def jacknifeError1D(rhos, avg=None):
Purpose:    calculate one jacknife error value
Parameters: 1D NumPy array RHOS representing bin values
            optional number AVG to replace the average of the given bin values
Returns:    number representing the jacknife error of RHOS

### def getPlateauFit(ycoords, yerr, min_slice, max_slice):
Purpose:    calculate plateau fit value for one data bin
Parameters: 1D NumPy array YCOORDS representing bin values for each timeslice
            1D NumPy array YERR representing jacknife error values for each timeslice
            number MIN_SLICE representing the start slice of the plateau fit
            number MAX_SLICE representing the end slice of the plateau fit
Returns:    plateau fit value over the given slices;
            None if the selected data contains junk values

### def plotPlateauFit(ycoords, yerr, min_slice, max_slice, delta=None, show=False):
Purpose:    calculate & display bin values, jacknife errors, and plateau fit
Parameters: 1D NumPy array YCOORDS representing bin values for each timeslice
            1D NumPy array YERR representing jacknife error values for each timeslice
            number MIN_SLICE representing the start slice of the plateau fit
            number MAX_SLICE representing the end slice of the plateau fit & plot
            optional number DELTA to manually set the error bars
Returns:    tuple representing the plateau fit value (index 0) and error (index 1);
            None if getPlateauFit() call fails

### def altFits1D(coords, err, static_bound, varied_bound, frac_delta=None, show=False):
Purpose:    calculate & display plateau fit starting points whose averages fall within the jacknife margin of error
Parameters: 1D NumPy array COORDS representing bin values for each timeslice
            1D NumPy array ERR representing jacknife error values for each timeslice
            number VARIED_BOUIND representing the boundary point whose alternatives will be calculated
            number STATIC_BOUND representing the boundary point (min or max slice) that will be held constant
            optional number FRAC_DELTA representing the minimum error window as a percentage of the average value
            optional boolean SHOW, if true alternate fit plots will display
Returns:    two-element list where element 0 is the final mean and element 1 is the error bound

### def altFits2D(binned, err, show=False, mode=0, selected_bins=None, length=3, max_fact=10, delta_fact=None):
Purpose:    wrapper function to call altFits1D() on multiple data bins
Parameters: 2D NumPy array BINNED where rows represent timeslices & columns represent bins
            1D NumPy array ERR representing jacknife error values for each timeslice
            optional list of bin indexes SELECTED_BINS to test, if None then test all
            optional integer LENGTH representing minimum length of the plateau
            optional number MAX_FACT representing the maximum error to plot as a multiple of the first slice's error
            optional boolean parameter INIT_PLATEAU, if true display plateau fit from index returned by minSlice()
            optional boolean parameters VARY_MIN, if true then vary the minimum slice index
            optional boolean parameters VARY_MAX, if true then vary the maximum slice index
            optional number DELTA_FACT representing the minimum error window as a percentage of the average value
Returns:    list storing non-null results of altFits1D() for the selected bins

### def getGroundState(data, show=False, mode=1, selected_bins=None, length=3, max_fact=10, delta_fact=0.05):
Purpose:    list ground state value & error from a raw data table
Parameters: 2D NumPy array representing the raw data, where rows represent timeslices and columns represent configurations
            optional boolean SHOW, toggles whether plots representing individual bins will display
            optional int MODE:
                mode 0 will set the minimum slice to the first plateau point and the maximum to the first point whose error is over MAX_FACT;
                mode 1 will lower the minimum slice until the average is outside the margin of error;
                mode 2 will raise the maximum slice until the average is outside the margin of error.
            optional list SELECTED_BINS representing the indeces of bins to test alternate fits for; if None, then test all
            optional int LENGTH, representing the minimum length of a plateau
            optional number MAX_FACT representing the error cutoff for the maximum slice as a multiple of the error for index 1 of its bin
            optional number DELTA_FACT, the minimum margin of error for testing alternate fits as a multiple of the average

### def listGroundStates(p, path, col_index, show=False, mode=1, selected_bins=None, length=3, max_fact=10, delta_fact=0.05):
Purpose:    wrapper function for getGroundState() to find ground states for multiple momentum values
Parameters: list of tuples P:
                element in the first position is an integer representing a momentum value;
                element in the second position is a list of string keywords associated with the files for that momentum value
            int COL_INDEX representing the index of the column storing the real part of the 2-point function
            for keyword args, see getGroundState()
Returns:    2D NumPy array where each row represents a momentum value:
                column 0 represents momentum;
                column 1 represents its average value;
                column 2 represents its jacknife error;
                columns 1 and 2 are set to None if data contains junk values or calculations failed

### def plotDispersion(vals_by_state, save_as):
Purpose:    plot and quadratic-fit a set of momenta with their average energies and error
Parameters: 2D NumPy array VALS_BY_STATE returned by listGroundStates()
            string SAVE_AS, path and file name where the dispersion relation plot will be saved
Returns:    void, dispersion relation plot and fit function saved to an external file
