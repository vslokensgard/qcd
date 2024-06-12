# Purpose:    plot and quadratic-fit a set of momenta with their average energies and error
# Parameters: 2D NumPy array VALS_BY_STATE returned by listGroundStates()
#             string SAVE_AS, path and file name where the dispersion relation plot will be saved
# Returns:    void, dispersion relation plot and fit function saved to an external file
def plotDispersion(vals_by_state, save_as):
    if not vals_by_state:
        return None
    for i in range(len(vals_by_state)):
        x[i], y[i], e[i] = vals_by_state[i]

    fit = np.polyfit(x, y, 2)
    fit_func = np.poly1d(fit)
    polyline = np.linspace(0, len(momenta), 100)
    plt.scatter(x, y)
    plt.errorbar(x, y, yerr=e, fmt="o")
    plt.plot(polyline, fit_func(polyline))
    plt.xlabel(str(fit_func))

    plt.savefig(save_as)
    return

# Purpose:    calculate & display bin values, jacknife errors, and plateau fit
# Parameters: 1D NumPy array YCOORDS representing bin values for each timeslice
#             1D NumPy array YERR representing jacknife error values for each timeslice
#             number MIN_SLICE representing the start slice of the plateau fit
#             number MAX_SLICE representing the end slice of the plateau fit & plot
#             optional number DELTA to manually set the error bars
# Returns:    tuple representing the plateau fit value (index 0) and error (index 1);
#             None if getPlateauFit() call fails
def plotPlateauFit(ycoords, yerr, min_slice, max_slice, delta=None, show=False):
    plat_fit = getPlateauFit(ycoords, yerr, min_slice, max_slice)
    if not plat_fit:
        return None

    if delta:
        plat_err = delta
    else:
        plat_err = jacknifeError1D(ycoords[min_slice:max_slice], avg=plat_fit)

    if show:
        title_fit = str(round(plat_fit, 5))
        title_err = str(round(plat_err, 5))
        plot_title = "Plateau fit value "+title_fit+" ± "+title_err
        plot_label = "Fit for slices "+str(min_slice)+" to "+str(max_slice)
        x = range(1, len(ycoords))
        y = ycoords[1:]
        e = yerr[1:]
        ax = plt.gca()
        ax.set_xlim([0, max_slice])
        point_range = np.abs(plat_fit - y[1])
        ymin = plat_fit - (point_range * 2)
        ymax = plat_fit + (point_range * 2)
        ax.set_ylim(ymin, ymax)

        plt.title(plot_title)
        plt.xlabel(plot_label)
        plt.scatter(x, y)
        plt.errorbar(x, y, yerr=e, fmt="o")
        plt.axvline(x=min_slice, linestyle="dashed")
        plt.axhline(y=plat_fit, linestyle="solid")
        plt.axhline(y=plat_fit-plat_err, linestyle=":")
        plt.axhline(y=plat_fit+plat_err, linestyle=":")
        plt.show()

    return [plat_fit, plat_err]

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
        plt.scatter(x, y)
        plt.errorbar(x, y, yerr=jack_error, fmt="o")
        plt.show()

    return