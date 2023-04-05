# This is the code for the Bloom Filter project of TDT4305

import configparser  # for reading the parameters file
from pathlib import Path  # for paths of files
import time  # for timing
# TODO import random prime

# Global parameters
parameter_file = 'default_parameters.ini'  # the main parameters file
data_main_directory = Path('data')  # the main path were all the data directories are
parameters_dictionary = dict()  # dictionary that holds the input parameters, key = parameter name, value = value
import sympy as sp

# DO NOT CHANGE THIS METHOD
# Reads the parameters of the project from the parameter file 'file'
# and stores them to the parameter dictionary 'parameters_dictionary'
def read_parameters():
    config = configparser.ConfigParser()
    config.read(parameter_file)
    for section in config.sections():
        for key in config[section]:
            if key == 'data':
                parameters_dictionary[key] = config[section][key]
            else:
                parameters_dictionary[key] = int(config[section][key])


# TASK 2
def bloom_filter(new_pass):

    n = parameters_dictionary["n"]

    return 0


# DO NOT CHANGE THIS METHOD
# Reads all the passwords one by one simulating a stream and calls the method bloom_filter(new_password)
# for each password read
def read_data(file):
    time_sum = 0
    pass_read = 0
    with file.open() as f:
        for line in f:
            pass_read += 1
            new_password = line[:-3]
            ts = time.time()
            bloom_filter(new_password)
            te = time.time()
            time_sum += te - ts

    return pass_read, time_sum


# TASK 1
# Created h number of hash functions
def hash_functions():

    hashes = []
    for key in parameters_dictionary.keys():
        print("Printing")
        print(key)
    h = parameters_dictionary["h"]
    n = parameters_dictionary["n"]

    for i in range(1, h+1):
        p_lower = i * 1000
        p_upper = (i + 1) * 1000
        p = sp.primerange(p_lower, p_upper)
        hashes.append(lambda S: sum(s * pow(p, i) for i, s in enumerate(S)) % n)

    return hashes


if __name__ == '__main__':
    # Reading the parameters
    read_parameters()

    # Creating the hash functions
    hash_functions()

    # Reading the data
    print("Stream reading...")
    data_file = (data_main_directory / parameters_dictionary['data']).with_suffix('.csv')
    passwords_read, times_sum = read_data(data_file)
    print(passwords_read, "passwords were read and processed in average", times_sum / passwords_read,
          "sec per password\n")
