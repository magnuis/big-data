# This is the code for the Bloom Filter project of TDT4305

import configparser  # for reading the parameters file
from pathlib import Path  # for paths of files
import time  # for timing
import sympy as sp
from bitarray import bitarray 


# Global parameters
parameter_file = 'default_parameters.ini'  # the main parameters file
data_main_directory = Path('data')  # the main path were all the data directories are
parameters_dictionary = dict()  # dictionary that holds the input parameters, key = parameter name, value = value

# the bloom filter and the hash functions must be global variables, in order to update and reuse it for every password read
filter = 0
hash_functions_list = []

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
    for hash_function in hash_functions_list:
        hash_value = hash_function(new_pass)
        filter[hash_value] = 1

# solution without bitarray
# def bloom_filter(new_pass):
#     for hash_function in hash_functions_list:
#         hash_value = hash_function(new_pass)
#         filter[hash_value] = 1

def reset_filter():
    '''
    :return: bitarray of length n with all values 0
    '''    
    n = parameters_dictionary["n"]
    new_filter = bitarray('0'*n)
    return new_filter

# solution without bitarray
# def reset_filter():
#     '''
#     :return: dictionary with n keys, each with value 0
#     '''    
#     n = parameters_dictionary["n"]
#     filter = dict()
#     for i in range(n):
#         filter[i] = 0
#     return filter


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


def hash_function(p, n):
    '''
    Generate a single hash function, as described in the task
    param p:
        prime number
    param n: 
        modulus
    return: 
        hash function to hash passwords
    '''
    return lambda S: sum(ord(s) * pow(p, i) for i, s in enumerate(S)) % n

def prime(i):
    '''
    Generate prime number
    param i: 
        lower bound factor
    return: 
        prime number between 100*i and 100*(i+1)
    '''
    p_lower = i * 100 
    p_upper = (i + 1) * 100
    return sp.randprime(p_lower, p_upper)


# TASK 1
# Created h number of hash functions
def hash_functions():

    hashes = []
    h = parameters_dictionary["h"]
    n = parameters_dictionary["n"]

    for i in range(h):
        p = prime(i) # make sure the prime number is different for each hash function
        hashes.append(hash_function(p, n))

    return hashes


def count_false_positives(passwords):
    '''
    Counts false negatives of a number of passwords passed through the bloom filter
    param passwords: 
        list of passwords that are not in the original password set
    return:
        number of false positives
    '''
    false_positives = 0
    for password in passwords:
        is_reported_seen = True
        for hash_function in hash_functions_list:
            hash_value = hash_function(password)
            # if one of the hashed values has a value of 1, the password is not priorly seen
            if filter[hash_value] == 0:
                is_reported_seen = False
                break

        if is_reported_seen:
            false_positives += 1

    return false_positives


# passwords from piazza
passwords_not_in_passwords_csv_file = ['07886819', '0EDABE59B', '0BFE0815B', '01058686E', '09DFB6D3F', '0F493202C', '0CA5E8F91', '0C13EC1D9', '05EF96537', '03948BA8F', '0D19FB394', '0BF3BD96C', '0D3665974', '0BBDF91E9', '0A6083B64', '0D76EF8EC', '096CD1830', '04000DE73', '025C442BA', '0FD6CAA0A', '06CC18905', '0998DDE00', '02BAACDC4', '0D58264FC', '0CB8911AA', '0CF9E0BDC', '007B7F82F', '0948FD17A', '058BB08DB', '02EDBE8CA', '0D6F02EFD', '09C9797FB', '0F8CB3DA5', '0C2825430', '038BE7E61', '03F69C0F5', '07EB08903', '0917C741D', '0D01FEE8F', '01B09A600', '0BD197525', '06B6A2E60', '0B72DEF61', '095B17373', '0B6E0EEB1', '0078B3053', '08BD9D53F', '01995361F', '0F0B50CAE', '0B5D2887E', '004EB658C', '0D2C77EDB', '07221E24D', '0E8A4CC90', '00E947367', '0DBE190BB', '0D8726592', '06C02D59D', '0462B8BC6', '0F85122F8', '0FA1961EB', '035230553', '04CDFB216', '0356DB0AD', '0FD947DA3', '053BB206F', '0D1772CC1', '00DB759F5', '072FB4E7A', '0B47CB62D', '0616B627F', '0F3E153BC', '0F3AC7DEE', '01286192B', '009F3C478', '07D89E83E', '007CAFDE6', '0ABC9E80B', '091D1CDA5', '0BFC208A1', '0957D4C84', '00AAF260A', '09CF00D7C', '0D1C66C72', '0EA20CA23', '07D6BE324', '05B264527', '0D48C41F6', '081E31BF5', '0A1DC7455', '07BB493D8', '050036F1B', '00E73A1EC', '0C2D93CC0', '0FF47B30C', '0313062DE', '0E1BEFA3F', '0A24D069F', '02A984386', '0367F7405']

if __name__ == '__main__':
    start_time = time.time()
    # Reading the parameters
    read_parameters()

    # Initializing the bloom filter
    filter = reset_filter()

    # Creating the hash functions
    hash_functions_list = hash_functions()

    # Reading the data
    print("Stream reading...")
    data_file = (data_main_directory / parameters_dictionary['data']).with_suffix('.csv')
    t = time.time()
    passwords_read, times_sum = read_data(data_file)
    print(f"read_data took {time.time() - t}, where {times_sum} seconds were used for bloom filter processing")
    print(passwords_read, "passwords were read and processed in average", times_sum / passwords_read,
          "sec per password\n")

    # Counting the number of false negatives
    false_positives = count_false_positives(passwords_not_in_passwords_csv_file)
    print(f"There were a total number of {false_positives} false positives out of {len(passwords_not_in_passwords_csv_file)} passwords \n")

    print("Total time:", time.time() - start_time)