# This is the code for the LSH project of TDT4305

import configparser  # for reading the parameters file
import sys  # for system errors and printouts
from pathlib import Path  # for paths of files
import os  # for reading the input data
import time  # for timing
from sympy import randprime  # for random prime number
from random import randint
from math import floor
import numpy as np

# Global parameters
parameter_file = 'default_parameters.ini'  # the main parameters file
data_main_directory = Path('data')  # the main path were all the data directories are
parameters_dictionary = dict()  # dictionary that holds the input parameters, key = parameter name, value = value
document_list = dict()  # dictionary of the input documents, key = document id, value = the document

all_shingles = []

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
            elif key == 'naive':
                parameters_dictionary[key] = bool(config[section][key])
            elif key == 't':
                parameters_dictionary[key] = float(config[section][key])
            else:
                parameters_dictionary[key] = int(config[section][key])


# DO NOT CHANGE THIS METHOD
# Reads all the documents in the 'data_path' and stores them in the dictionary 'document_list'
def read_data(data_path):
    for (root, dirs, file) in os.walk(data_path):
        for f in file:
            file_path = data_path / f
            doc = open(file_path).read().strip().replace('\n', ' ')
            file_id = int(file_path.stem)
            document_list[file_id] = doc


# DO NOT CHANGE THIS METHOD
# Calculates the Jaccard Similarity between two documents represented as sets
def jaccard(doc1, doc2):
    return len(doc1.intersection(doc2)) / float(len(doc1.union(doc2)))


# DO NOT CHANGE THIS METHOD
# Define a function to map a 2D matrix coordinate into a 1D index.
def get_triangle_index(i, j, length):
    if i == j:  # that's an error.
        sys.stderr.write("Can't access triangle matrix with i == j")
        sys.exit(1)
    if j < i:  # just swap the values.
        temp = i
        i = j
        j = temp

    # Calculate the index within the triangular array. Taken from pg. 211 of:
    # http://infolab.stanford.edu/~ullman/mmds/ch6.pdf
    # adapted for a 0-based index.
    k = int(i * (length - (i + 1) / 2.0) + j - i) - 1

    return k


# DO NOT CHANGE THIS METHOD
# Calculates the similarities of all the combinations of documents and returns the similarity triangular matrix
def naive():
    docs_Sets = []  # holds the set of words of each document

    for doc in document_list.values():
        docs_Sets.append(set(doc.split()))

    # Using triangular array to store the similarities, avoiding half size and similarities of i==j
    num_elems = int(len(docs_Sets) * (len(docs_Sets) - 1) / 2)
    similarity_matrix = [0 for x in range(num_elems)]
    for i in range(len(docs_Sets)):
        for j in range(i + 1, len(docs_Sets)):
            similarity_matrix[get_triangle_index(i, j, len(docs_Sets))] = jaccard(docs_Sets[i], docs_Sets[j])

    return similarity_matrix


# METHOD FOR TASK 1
# Creates the k-Shingles of each document and returns a list of them
def k_shingles():
    k = parameters_dictionary['k']
    docs_k_shingles = []  # holds the k-shingles of each document

    for doc in document_list.values():
        doc_shingle = set()
        for i in range(len(doc) - k + 1):
            doc_shingle.add(doc[i:i + k])
        docs_k_shingles.append(list(doc_shingle))

    #print(list(docs_k_shingles))
    return list(docs_k_shingles)


# METHOD FOR TASK 2
# Creates a signatures set of the documents from the k-shingles list
def signature_set(k_shingles):
    '''
    :param k_shingles: list of k-shingles of each document
    :return: list of signature sets of each document
    '''
    docs_sig_sets = []

    shingle_array = np.array(k_shingles[0])

    for k_shingle in k_shingles:
        shingle_array = np.union1d(shingle_array, np.array(k_shingle))
    
    t0 = time.time()
    count = 0
    for shingle in k_shingles:
        signature = np.zeros(shingle_array.size, dtype=np.int8)
        shingle = np.array(shingle)
        count += 1
        if (count % 50 == 0):
            print(str(count) + " docs at time " + str(time.time() - t0) + " seconds")
        for i in range(signature.size):
            if shingle_array[i] in shingle:
                signature[i] = 1
        docs_sig_sets.append(list(signature))

    return docs_sig_sets


def hash(a, b, p, N):
    return lambda x: ((a * x + b) % p) % N

# METHOD FOR TASK 3
# Creates the minHash signatures after simulation of permutations
def minHash(docs_signature_sets):
    min_hash_signatures = []
    
    permutations = parameters_dictionary['permutations']
    N = len(docs_signature_sets[0])

    min_hash_signatures = [[] for i in range(len(docs_signature_sets))]
    # one loop per k permutations
    for _ in range(permutations):
        a = randint(0, N)   
        b = randint(0, N)
        p = randprime(N, N**2) 
        
        # loop over all document signatures
        for signature_index, signature_set in enumerate(docs_signature_sets):
            min_hash = N
            # loop over all shingles
            signature_set = np.array(signature_set, dtype=np.int8)
            for shingle_index, shingle in enumerate(signature_set):
                if shingle == 1:
                    hash_value = hash(a, b, p, N)(shingle_index)
                    min_hash = min(min_hash, hash_value)
            min_hash_signatures[signature_index].append(min_hash)

    return min_hash_signatures


# METHOD FOR TASK 4
# Hashes the MinHash Signature Matrix into buckets and find candidate similar documents
def lsh(m_matrix):
    candidates = []  # list of candidate sets of documents for checking similarity
    
    buckets = parameters_dictionary['buckets']
    r = parameters_dictionary['r']

    m_matrix_banded = []

    for signature_index, signature in enumerate(m_matrix):
        assert len(signature) % r == 0 # <-- the number of entries must be divisible by r
        banded_signature = []
        for i in range(0, len(signature), r):
            band = (signature[i:i+r])
            banded_signature.append(band)
        m_matrix_banded.append(np.array(banded_signature))

    # one loop per band
    for band_index in range(len(m_matrix_banded[0])):   
        hash_buckets = {}
        # one loop per signature in the band
        for signature_index, signature in enumerate(m_matrix_banded):
            # simple hashing, taking the sum of the band and modding it by the number of buckets
            band = sum(signature[band_index])
            hash_value = (band) % buckets
            # add hash value into corresponding bucket
            if hash_value not in hash_buckets:
                hash_buckets[hash_value] = [signature_index]
            else:
                hash_buckets[hash_value].append(signature_index)


        # check for collisions
        for bucket in hash_buckets.values():
            if len(bucket) > 1: # <-- not consider buckets with one signature
                for i in range(len(bucket)-1):
                    candidate_pair = (bucket[i], bucket[i+1])
                    if candidate_pair not in candidates:
                        candidates.append(candidate_pair)
    # print(m_matrix)
    # print(m_matrix_banded)
    # print(candidates)
    print(candidates)
    return candidates


def similarity(doc1, doc2):
    permutations = parameters_dictionary['permutations']
    for i in range(len(doc1)):
        count = 0
        if doc1[i] == doc2[i]:
            count += 1
    return count / permutations

# METHOD FOR TASK 5
# Calculates the similarities of the candidate documents
def candidates_similarities(candidate_docs, min_hash_matrix):
    similarity_matrix = []

    for pair in candidate_docs:
        # doc1 = set(min_hash_matrix[pair[0]])
        # doc2 = set(min_hash_matrix[pair[1]])
        # similarity = jaccard(doc1=doc1, doc2=doc2)

        hash_1 = min_hash_matrix[pair[0]]
        hash_2 = min_hash_matrix[pair[1]]
        
        sim = similarity(hash_1, hash_2)
        
        similarity_matrix.append(((pair[0], pair[1]), sim))

    return similarity_matrix


# METHOD FOR TASK 6
# Returns the document pairs of over t% similarity
def return_results(lsh_similarity_matrix):
    document_pairs = []
    t = parameters_dictionary['t']
    count = 0
    for row in lsh_similarity_matrix:
        if row[1] >= t:
            document_pairs.append(row[0])
        else:
            count += 1
    print("Number of false positives: " + str(count) + " out of " + str(len(lsh_similarity_matrix)) + " candidates.")
    return document_pairs


# METHOD FOR TASK 6
def count_false_neg_and_pos(lsh_similarity_matrix, naive_similarity_matrix):
    false_negatives = 0
    false_positives = 0

    length = len(document_list)
    t = parameters_dictionary['t']

    for candidate_pair in lsh_similarity_matrix:
        i = candidate_pair[0][0]
        j = candidate_pair[0][1]
        triangle_index = get_triangle_index(i, j, length)
       # print("for i = " + str(i) + " and j = " + str(j) + " the similarity is " + str(naive_similarity_matrix[triangle_index]))
        
        naive_similarity = naive_similarity_matrix[triangle_index]
        candidate_similarity = candidate_pair[1]

        if naive_similarity >= t and candidate_similarity < t:
            false_positives += 1
        elif naive_similarity < t and candidate_similarity >= t:
            false_negatives += 1

    return false_negatives, false_positives


# DO NOT CHANGE THIS METHOD
# The main method where all code starts
if __name__ == '__main__':
    # Reading the parameters
    read_parameters()

    # Reading the data
    print("Data reading...")
    data_folder = data_main_directory / parameters_dictionary['data']
    t0 = time.time()
    read_data(data_folder)
    document_list = {k: document_list[k] for k in sorted(document_list)}
    t1 = time.time()
    print(len(document_list), "documents were read in", t1 - t0, "sec\n")

    # Naive
    naive_similarity_matrix = []
    if parameters_dictionary['naive']:
        print("Starting to calculate the similarities of documents...")
        t2 = time.time()
        naive_similarity_matrix = naive()
        t3 = time.time()
        print("Calculating the similarities of", len(naive_similarity_matrix),
              "combinations of documents took", t3 - t2, "sec\n")

    # k-Shingles
    print("Starting to create all k-shingles of the documents...")
    t4 = time.time()
    all_docs_k_shingles = k_shingles()
    t5 = time.time()
    print("Representing documents with k-shingles took", t5 - t4, "sec\n")

    # signatures sets
    print("Starting to create the signatures of the documents...")
    t6 = time.time()
    signature_sets = signature_set(all_docs_k_shingles)
    t7 = time.time()
    print("Signatures representation took", t7 - t6, "sec\n")

    # Permutations
    print("Starting to simulate the MinHash Signature Matrix...")
    t8 = time.time()
    min_hash_signatures = minHash(signature_sets)
    t9 = time.time()
    print("Simulation of MinHash Signature Matrix took", t9 - t8, "sec\n")

    # LSH
    print("Starting the Locality-Sensitive Hashing...")
    t10 = time.time()
    candidate_docs = lsh(min_hash_signatures)
    t11 = time.time()
    print("LSH took", t11 - t10, "sec\n")

    # Candidate similarities
    print("Starting to calculate similarities of the candidate documents...")
    t12 = time.time()
    lsh_similarity_matrix = candidates_similarities(candidate_docs, min_hash_signatures)
    t13 = time.time()
    print("Candidate documents similarity calculation took", t13 - t12, "sec\n\n")

    # Return the over t similar pairs
    print("Starting to get the pairs of documents with over ", parameters_dictionary['t'], "% similarity...")
    t14 = time.time()
    pairs = return_results(lsh_similarity_matrix)
    t15 = time.time()
    print("The pairs of documents are:\n")
    for p in pairs:
        print(p)
    print("\n")

    # Count false negatives and positives
    if parameters_dictionary['naive']:
        print("Starting to calculate the false negatives and positives...")
        t16 = time.time()
        false_negatives, false_positives = count_false_neg_and_pos(lsh_similarity_matrix, naive_similarity_matrix)
        t17 = time.time()
        print("False negatives = ", false_negatives, "\nFalse positives = ", false_positives, "\n\n")

    if parameters_dictionary['naive']:
        print("Naive similarity calculation took", t3 - t2, "sec")

    print("LSH process took in total", t13 - t4, "sec")



def hash(a: int, b: int, p: int, N: int, x: int):
    return ((a * x + b) % p) % N





