import numpy as np
import os

junk_flag = False
time_slices = 64

# Purpose:    extract file names containing keyword strings from a directory
# Parameters: list of keyword strings KEYWORDS
#             optional directory path string PATH (must end in a "/")
#             if no PATH given, searches the current directory
# Returns:    list of file name strings matching keywords in directory
def searchDir(keywords, path=None):
    if path:
        all_files = [(path + k) for k in os.listdir(path)]

    else:
        all_files = os.listdir()

    subset = [k for k in all_files if any(keyword in k for keyword in keywords)]
    return subset

# Purpose:    read list of input files to single data table
# Parameters: list of file names to read FILE_LIST;
#             index of selected column COL_INDEX
# Returns:    NumPy table of input data
#             Columns represent input files
#             Rows represent time slices
def mergeTables(file_list, col_index):
        by_file = np.zeros((len(file_list),time_slices))
        rows_to_delete = []

        for i in range(len(file_list)):
            current_file = np.genfromtxt(file_list[i])
            by_param = np.transpose(current_file)
            current_row = by_param[col_index]

            if len(current_row) != time_slices:
                junk_flag = True
                rows_to_delete.append(i)
                print("Selected column of file",i,"has length",len(current_row))
                continue

            current_row = np.nan_to_num(current_row)
            if np.all(current_row):
                by_file[i] = current_row
            else:
                junk_flag = True
                rows_to_delete.append(i)
                print("Selected column of file",i,"contains junk values")
                continue

        if rows_to_delete:
            to_delete = np.array(rows_to_delete)
            by_file = np.delete(by_file, to_delete, axis=0)
            print("Invalid configurations deleted")

        by_slice = np.transpose(by_file)
        if np.all(by_slice):
            return by_slice
        else:
            print("Encountered unexpected configurations, exiting")
            exit()

# Purpose:    calculate table of jacknife bins for input data
# Parameters: NumPy table DATA:
#             Columns represent trials
#             Rows represent time slices
# Returns:   NumPy table
#             Columns represent jacknife bins
#             Rows represent time slices
def jacknifeAverage(data):
    binned = np.zeros(data.shape)
    num_slices, num_bins = data.shape

    for i in range(num_slices):
        row_sum = np.sum(data[i])
        for j in range(num_bins):
            binned[i][j] = (row_sum - data[i][j]) / (num_bins - 1)

    return binned

# Purpose:    calculate effective energy from 2-point function values
# Parameters: NumPy table BINS:
#             Columns represent jacknife bin values of 2-point function
#             Rows represent time slices
# Returns:    NumPy table
#             Columns represent jacknife bin values for effective energy
#             Rows represent time slices
def effectiveEnergy(bins, debug=False):
    if debug: print("effectiveEnergy()")
    num_intervals = bins.shape[0] - 1
    num_bins = bins.shape[1]
    in_by_bin = np.transpose(bins)
    out_by_bin = np.zeros((num_bins, num_intervals))
    bins_to_delete = []

    for bin_no in range(num_bins):

        for int_no in range(num_intervals):
            num = in_by_bin[bin_no][int_no]
            den = in_by_bin[bin_no][int_no+1]
            if den == 0:
                print("Tried to divide by zero on bin no.",bin_no,"exiting")
                exit()
            quo = num / den
            if quo <= 0:
                print("Could not take log for bin no.",bin_no,"on interval",int_no)
                out_by_bin[bin_no][int_no] = 0
            else:
                out_by_bin[bin_no][int_no] = np.log(in_by_bin[bin_no][int_no] / in_by_bin[bin_no][int_no+1])

    out_by_slice = np.transpose(out_by_bin)
    return out_by_slice

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

# Purpose:    calculate jacknife error across all bins for each timeslice
# Parameters: NumPy table BINNED_DATA:
#             rows representing timeslices,
#             columns representing bins
# Returns:    NumPy 1D array:
#             one error value per timeslice
def jacknifeError2D(binned_data, rho=None, debug=False):
    num_slices, num_bins = binned_data.shape
    bins_used = num_bins
    delta_rho = np.zeros(num_slices)
    if rho:
        rho_bars = rho
    else:
        rho_bars = averageColumns(binned_data)

    for slice_no in range(num_slices):
        rho_bar = rho_bars[slice_no]
        sigma = 0

        for bin_no in range(num_bins):
            if binned_data[slice_no][bin_no] == 0:
                if debug: print("Skipping invalid entry for slice",slice_no,"on bin",bin_no)
                bins_used -= 1
                continue
            diff = binned_data[slice_no][bin_no] - rho_bar
            sigma += np.square(diff)

        if bins_used == 0:
            print("No valid data for slice",slice_no)
            delta_rho[slice_no] = 0
        else:
            m_factor = (bins_used - 1) / bins_used
            delta_rho[slice_no] = np.sqrt(sigma * m_factor)

    return delta_rho

# Purpose:    identify the first timeslice whose error exceeds a given threshold
# Parameters: 1D Numpy array of error values JACKNIFE_ERR
#             number MAX_FACT, product with the error of index 1 of JACKNIFE_ERR will set the threshold
# Returns:    index of first value in JACKNIFE_ERR greater than the threshold, or 5, whichever is larger
def maxSlice(jacknife_err, max_fact):
    starting_length = len(jacknife_err)
    err_threshold = jacknife_err[1] * max_fact

    for i in range(1, starting_length):
        if not jacknife_err[i]:
            i -=1
            break
        if jacknife_err[i] > err_threshold:
            break

    if i < 5:
        i = 5

    #return i
    return 16

# Purpose:    graph effective energy values with jacknife error for each bin
# Parameters: 2D NumPy array EFF_ENERGIES:
#             rows represent timeslices,
#             columns represent bins;
#             1D NumPy array JACK_ERROR:
#             represents error values associated with each bin value
#             number ERR_CUTOFF representing max error as a factor of the error of JACK_ERROR[1]
#             list of integers SELECTED_BINS, if not None then only graph listed indeces
# Returns:    void
def jacknifeGraph(eff_energies, jack_error, err_cutoff=None, selected_bins=None):
    energies_by_bin = np.transpose(eff_energies)
    num_bins, num_intervals = energies_by_bin.shape
    min_slice = 1
    if err_cutoff:
        max_slice = maxSlice(jack_error, err_cutoff)
    else:
        max_slice = num_intervals

    x = range(min_slice, max_slice)
    jack_error = jack_error[min_slice:max_slice]

    if selected_bins:
        to_plot = selected_bins
    else:
        to_plot = list(range(num_bins))

    for bin_no in to_plot:
        y = energies_by_bin[bin_no][min_slice:max_slice]
        #plt.scatter(x, y)
        #plt.errorbar(x, y, yerr=jack_error, fmt="o")
        #plt.show()

    return

# Purpose:    identify the start indexes of plateaus of a given length, excluding index 0
# Parameters: 1D NumPy array representing the bin values for each timeslice YCOORDS
#             1D NumPy array representing jacknife errors for each timeslice YERR
#             optional number LENGTH representing minimum length of the plateau
# Returns:    start index of the plateau, 0 if no plateau is found
def minSlice(ycoords, yerr, length=3):
    num_coords = len(ycoords)
    i = 1
    count = 0
    max_err = 0
    min_err = 0

    while i < num_coords:
        if count == length:
            return i - count
        if (not ycoords[i]) or (not yerr[i]):
            return i - count
        else:
            if (max_err - min_err) > 0:
                curr_max = ycoords[i] + yerr[i]
                curr_min = ycoords[i] - yerr[i]
                if max_err > curr_max:
                    max_err = curr_max
                if min_err < curr_min:
                    min_err = curr_min
                i += 1
                count += 1

            else:
                i -= count
                i += 1

                if i >= num_coords:
                    return 0

                curr_max = ycoords[i] + yerr[i]
                curr_min = ycoords[i] - yerr[i]
                max_err = curr_max
                min_err = curr_min
                count = 0

    return 0

# Purpose:    wrapper function to run multiple bins of data through minSlice()
# Parameters: 2D NumPy array of timeslice rows & bin columns BINNED_VALS
#             1D NumPy array representing jacknife errors for each timeslice BINNED_ERR
#             optional number LENGTH representing minimum length of the plateau
#             optional list of bin indexes SELECTED_BINS to test, if None then test all
# Returns:    1D NumPy array of the plateau start index for each selected bin
def listPlateaus(binned_vals, binned_err, length=3, selected_bins=None):
    vals_by_bin = np.transpose(binned_vals)
    if selected_bins:
        to_plot = selected_bins
    else:
        to_plot = range(len(vals_by_bin))
    plateaus = np.zeros(len(to_plot))

    for bin_no in to_plot:
        plateaus[bin_no] = minSlice(vals_by_bin[bin_no], binned_err, length=length)

    return plateaus

# Purpose:    calculate one jacknife error value
# Parameters: 1D NumPy array RHOS representing bin values
#             optional number AVG to replace the average of the given bin values
# Returns:    number representing the jacknife error of RHOS
def jacknifeError1D(rhos, avg=None):
    if avg:
        rho_bar = avg
    else:
        rho_bar = np.mean(rhos)

    m_fact = (len(rhos) - 1) / len(rhos)
    sig_fact = np.sum( np.square(rhos - rho_bar) )
    delta_t = np.sqrt(m_fact * sig_fact)

    return delta_t

# Purpose:    calculate plateau fit value for one data bin
# Parameters: 1D NumPy array YCOORDS representing bin values for each timeslice
#             1D NumPy array YERR representing jacknife error values for each timeslice
#             number MIN_SLICE representing the start slice of the plateau fit
#             number MAX_SLICE representing the end slice of the plateau fit
# Returns:    plateau fit value over the given slices;
#             None if the selected data contains junk values
def getPlateauFit(ycoords, yerr, min_slice, max_slice):
    if not np.all(yerr[min_slice:max_slice]):
        print("No calculable fit for indexes",min_slice,"to",max_slice)
        return 0
    num = np.sum(ycoords[min_slice:max_slice] * np.power(yerr[min_slice:max_slice], -2))
    den = np.sum(np.power(yerr[min_slice:max_slice], -2))
    if not den:
        print("No calculable fit for indexes",min_slice,"to",max_slice)
        return 0
    quo = num/den
    return quo

# Purpose:    calculate & display plateau fit starting points whose averages fall within the jacknife margin of error
# Parameters: 1D NumPy array COORDS representing bin values for each timeslice
#             1D NumPy array ERR representing jacknife error values for each timeslice
#             number VARIED_BOUIND representing the boundary point whose alternatives will be calculated
#             number STATIC_BOUND representing the boundary point (min or max slice) that will be held constant
#             optional number FRAC_DELTA representing the minimum error window as a percentage of the average value
#             optional boolean SHOW, if true alternate fit plots will display
# Returns:    two-element list where element 0 is the final mean and element 1 is the error bound
def altFits1D(coords, err, static_bound, varied_bound, frac_delta=None, show=False):
    if varied_bound <= static_bound:
        rho_bar = getPlateauFit(coords, err, varied_bound, static_bound)
        if not rho_bar:
            return 0
        delta_e = jacknifeError1D(coords[varied_bound:static_bound], avg=rho_bar)
        if frac_delta:
            if (delta_e/rho_bar < frac_delta):
                bound = rho_bar * frac_delta
        else:
            bound = delta_e

        while varied_bound > 0:
            next_slice = varied_bound - 1
            include_next_plateau = getPlateauFit(coords, err, next_slice, static_bound)
            if include_next_plateau <= 0:
                return [rho_bar, bound]

            elif (rho_bar-bound) <= include_next_plateau <= (rho_bar+bound):
                varied_bound = next_slice
                rho_bar = include_next_plateau

            else:
                return [rho_bar, bound]

    else:
        rho_bar = getPlateauFit(coords, err, static_bound, varied_bound)
        if not rho_bar:
            return 0
        delta_e = jacknifeError1D(coords[varied_bound:static_bound], avg=rho_bar)
        if (delta_e/rho_bar < frac_delta):
            bound = rho_bar * frac_delta
        else:
            bound = delta_e

        while varied_bound < len(coords):
            next_slice = varied_bound + 1
            include_next_plateau = getPlateauFit(coords, err, static_bound, next_slice)

            if include_next_plateau <= 0:
                return [rho_bar, bound]
            elif (rho_bar-bound) <= include_next_plateau <= (rho_bar+bound):
                varied_bound = next_slice
                rho_bar = include_next_plateau
            else:
                return [rho_bar, bound]

# Purpose:    wrapper function to call altFits1D() on multiple data bins
# Parameters: 2D NumPy array BINNED where rows represent timeslices & columns represent bins
#             1D NumPy array ERR representing jacknife error values for each timeslice
#             optional list of bin indexes SELECTED_BINS to test, if None then test all
#             optional integer LENGTH representing minimum length of the plateau
#             optional number MAX_FACT representing the maximum error to plot as a multiple of the first slice's error
#             optional boolean parameter INIT_PLATEAU, if true display plateau fit from index returned by minSlice()
#             optional boolean parameters VARY_MIN, if true then vary the minimum slice index
#             optional boolean parameters VARY_MAX, if true then vary the maximum slice index
#             optional number DELTA_FACT representing the minimum error window as a percentage of the average value
# Returns:    list storing non-null results of altFits1D() for the selected bins
def altFits2D(binned, err, debug=False, show=False, mode=1, selected_bins=None, length=3, max_fact=10, delta_fact=None):
    if debug: print("altFits2D()")
    sort_by_bin = np.transpose(binned)
    fit_vals = []

    if selected_bins:
        to_plot = selected_bins
    else:
        to_plot = range(len(sort_by_bin))

    for bin_no in to_plot:
        bin_data = sort_by_bin[bin_no]
        plateau_min = minSlice(bin_data, err, length=length)
        plateau_max = maxSlice(err, max_fact=max_fact)

        if mode == 0:
            average = fit_vals.append(getPlateauFit(bin_data, err, plateau_min, plateau_max))
            if average: fit_vals.append(average)
        if mode == 1:
            average = altFits1D(bin_data, err, plateau_max, plateau_min, delta_fact, show=show)[0]
            if average: fit_vals.append(average)
        if mode == 2:
            average = altFits1D(bin_data, err, plateau_min, plateau_max, delta_fact, show=show)[0]
            if average: fit_vals.append(average)

    if fit_vals:
        if debug: print("Successfully calculated plateau fits")
        fit_vals = np.array(fit_vals)
    else:
        if debug: print("Error calculating plateau fits, exiting")
        exit()

    return fit_vals

# Purpose:    list ground state value & error from a raw data table
# Parameters: 2D NumPy array representing the raw data, where rows represent timeslices and columns represent configurations
#             optional boolean SHOW, toggles whether plots representing individual bins will display
#             optional int MODE:
#                 mode 0 will set the minimum slice to the first plateau point and the maximum to the first point whose error is over MAX_FACT;
#                 mode 1 will lower the minimum slice until the average is outside the margin of error;
#                 mode 2 will raise the maximum slice until the average is outside the margin of error.
#             optional list SELECTED_BINS representing the indeces of bins to test alternate fits for; if None, then test all
#             optional int LENGTH, representing the minimum length of a plateau
#             optional number MAX_FACT representing the error cutoff for the maximum slice as a multiple of the error for index 1 of its bin
#             optional number DELTA_FACT, the minimum margin of error for testing alternate fits as a multiple of the average
def getGroundState(data, debug=False, show=False, mode=1, selected_bins=None, length=3, max_fact=10, delta_fact=0.05):
    if debug: print("getGroundState()")
    twoptf = np.nan_to_num(jacknifeAverage(data))
    if np.all(twoptf):
        if debug: print("All jacknife bin values are nonzero")
    else:
        print("Jacknife bins contain zero values, exiting")
        exit()

    energy = np.nan_to_num(effectiveEnergy(twoptf, debug=False))
    if np.all(energy):
        if debug: print("All effective energy values are valid")
    else:
        print("Invalid effective energy entries")

    err = np.nan_to_num(jacknifeError2D(energy, debug=True))
    if np.all(err):
        if debug: print("Jacknife error values are valid")
    else:
        print("Invalid jacknife error values")

    vals = altFits2D(energy, err, show=show, mode=mode, selected_bins=selected_bins, length=length, max_fact=max_fact, delta_fact=delta_fact)
    val_array = np.array(vals)
    avg = np.mean(val_array)
    dev = jacknifeError1D(val_array, avg=avg)

    return avg, dev
#    return

# Purpose:    wrapper function for getGroundState() to find ground states for multiple momentum values
# Parameters: list of tuples P:
#                 element in the first position is an integer representing a momentum value;
#                 element in the second position is a list of string keywords associated with the files for that momentum value
#             int COL_INDEX representing the index of the column storing the real part of the 2-point function
#             for keyword args, see getGroundState()
# Returns:    2D NumPy array where each row represents a momentum value:
#                 column 0 represents momentum;
#                 column 1 represents its average value;
#                 column 2 represents its jacknife error;
#                 columns 1 and 2 are set to None if data contains junk values or calculations failed
def listGroundStates(p, path, col_index, debug=False, show=False, mode=1, selected_bins=None, length=3, max_fact=10, delta_fact=0.05):
    ground_states = np.zeros((len(p),3))

    for i in range(len(p)):
        file_list = searchDir(p[i][1], path=path)
        if debug: print(file_list,"end files for momentum",p[i][0])
        data_table = mergeTables(file_list, col_index)
        ground_states[i][0] = p[i][0]
        avg, dev = getGroundState(data_table, debug=True, mode=mode, delta_fact=delta_fact)

        if avg and dev:
            ground_states[i][1] = avg
            ground_states[i][2] = dev
        else:
            ground_states[i][1] = None
            ground_states[i][2] = None
        np.savetxt("ground-states", ground_states)

    return avg, dev

# Purpose:    plot and quadratic-fit a set of momenta with their average energies and error
# Parameters: 2D NumPy array VALS_BY_STATE returned by listGroundStates()
#             string SAVE_AS, path and file name where the dispersion relation plot will be saved
# Returns:    void, dispersion relation plot and fit function saved to an external file
#def plotDispersion(vals_by_state, save_as):
#    if not vals_by_state:
#        return None
#    for i in range(len(vals_by_state)):
#        x[i], y[i], e[i] = vals_by_state[i]
#
#    fit = np.polyfit(x, y, 2)
#    fit_func = np.poly1d(fit)
#    polyline = np.linspace(0, len(momenta), 100)
#    plt.scatter(x, y)
#    plt.errorbar(x, y, yerr=e, fmt="o")
#    plt.plot(polyline, fit_func(polyline))
#    plt.xlabel(str(fit_func))
#
#    plt.savefig(save_as)
#    return
