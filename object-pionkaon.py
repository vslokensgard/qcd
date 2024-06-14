import numpy as np
import os

time_slices = 64

# UTILITIES

def searchDir(keywords, path=None):
    if path:
        all_files = [(path + k) for k in os.listdir(path)]

    else:
        all_files = os.listdir()

    subset = [k for k in all_files if any(keyword in k for keyword in keywords)]
    return subset

def mergeTables(file_list, col_index, debug=False):
        by_file = np.zeros((len(file_list),time_slices))
        rows_to_delete = []
        for i in range(len(file_list)):
            current_file = np.genfromtxt(file_list[i])
            by_param = np.transpose(current_file)
            current_row = by_param[col_index]

            if len(current_row) != time_slices:
                rows_to_delete.append(i)
                if debug: print("Selected column of file",i,"has length",len(current_row))
                continue

            current_row = np.nan_to_num(current_row)
            if np.all(current_row):
                by_file[i] = current_row
            else:
                rows_to_delete.append(i)
                if debug: print("Selected column of file",i,"contains junk values")
                continue

        if rows_to_delete:
            to_delete = np.array(rows_to_delete)
            by_file = np.delete(by_file, to_delete, axis=0)
            if debug: print("Invalid configurations deleted")

        by_slice = np.transpose(by_file)
        if np.all(by_slice):
            return by_slice
        else:
            print("Encountered unexpected configurations, exiting")
            exit()

def jacknifeBins(data):
    binned = np.zeros(data.shape)
    num_slices, num_bins = data.shape

    for i in range(num_slices):
        row_sum = np.sum(data[i])
        for j in range(num_bins):
            binned[i][j] = (row_sum - data[i][j]) / (num_bins - 1)

    return binned

def averageColumns(vals):
    num_intervals, num_trials = vals.shape
    avg = np.sum(vals, axis=1)
    avg /= num_trials

    return avg

def jacknifeError1D(rhos, avg=None):
    if avg:
        rho_bar = avg
    else:
        rho_bar = np.mean(rhos)

    m_fact = (len(rhos) - 1) / len(rhos)
    sig_fact = np.sum( np.square(rhos - rho_bar) )
    delta_t = np.sqrt(m_fact * sig_fact)

    return delta_t

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
            if debug: print("No valid data for slice",slice_no)
            delta_rho[slice_no] = 0
        else:
            m_factor = (bins_used - 1) / bins_used
            delta_rho[slice_no] = np.sqrt(sigma * m_factor)

    return delta_rho

# ENERGY

def Energy:
    debug = False
    
    def __init__(self, bins):
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
                    if debug: print("Could not take log for bin no.",bin_no,"on interval",int_no)
                    out_by_bin[bin_no][int_no] = 0
                else:
                    out_by_bin[bin_no][int_no] = np.log(in_by_bin[bin_no][int_no] / in_by_bin[bin_no][int_no+1])

        out_by_slice = np.transpose(out_by_bin)
        
        self.by_bin = out_by_bin
        self.by_slice = out_by_slice
        self.num_bins = len(out_by_bin)
        self.num_slices = len(out_by_slice)
        self.errors = jacknifeError2D(out_by_slice)
        return

def Plateau:
    debug = False
    
    def __init__(self, energies, max_err_fact=10, min_plateau=3, set_min=None, set_max=None):
        if set_min:
            self.min_slices = np.full(energies.num_bins, set_min)
        else:
            self.min_slices = listMinSlices(energies, min_plateau=min_plateau)

        if set_max:
            self.max_slices = np.full(energies.num_bins, set_max)
        else:
            self.max_slices = np.full(energies.num_bins, max_slice(energies.errors, max_err_fact) )

        self.default_fits = np.zeros(energies.num_bins)
        for i in range(energies.num_bins):
            default_fits[i] = getPlateauFit(energies.by_bin[i], energies.errors, energies.min_slices[i], energies.max_slices[i])

    def maxSlice(jacknife_err, max_fact):
        starting_length = len(jacknife_err)
        err_threshold = jacknife_err[1] * max_fact

        for i in range(1, starting_length):
            if not jacknife_err[i]:
                i -= 1
                break
            if jacknife_err[i] > err_threshold:
                break

        return i

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

    def listMinSlices(energies, min_plateau):
        coords = energies.by_bin
        errors = energies.errors
        min_list = np.zeros( (energies.num_bins) )
    
        for i in range(energies.num_bins):
            min_list[i] = minSlice(coords[i], errors, length=min_plateau)

        return min_list

    def getPlateauFit(ycoords, yerr, min_slice, max_slice, debug=True):
        if not np.all(yerr[min_slice:max_slice]):
            if debug: print("No calculable fit for indexes",min_slice,"to",max_slice)
            return 0
        num = np.sum(ycoords[min_slice:max_slice] * np.power(yerr[min_slice:max_slice], -2))
        den = np.sum(np.power(yerr[min_slice:max_slice], -2))
        if not den:
            if debug: print("No calculable fit for indexes",min_slice,"to",max_slice)
            return 0
        quo = num/den
        return quo