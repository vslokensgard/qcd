import os
import numpy as np

def momenta(path, keywords, destination, slices=64):
    directories = os.listdir(path)

    for d in directories:
        dir_path = path + d + "/"
        #print(dir_path)
        dir_files = os.listdir(dir_path)
        #print(dir_files)
        to_average = [k for k in dir_files if any(keyword in k for keyword in keywords)]
        tables = []
        for a in to_average:
            address = dir_path + a
            tables.append(np.genfromtxt(address) / len(to_average))

        if tables:
            avg_table = np.zeros(tables[0].shape)
            for t in tables:
                avg_table += (t / len(tables))

            np.savetxt( (destination + d), avg_table)

        else:
            continue

    return

def splitFiles(filenames, path=None, label="energy", lower_bound=0, upper_bound=64):

    for fn in filenames:

        if path:
            address = path + fn
        else:
            address = fn

        full_table = np.genfromtxt(address)
        cropped_table = full_table[lower_bound:upper_bound]
        np.savetxt( (label + fn), cropped_table)

    return
