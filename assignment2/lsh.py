# This is the code for the LSH project of TDT4305

import configparser
import random  # for reading the parameters file
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
    '''
    :return: list of k-shingles of each document
    '''
    k = parameters_dictionary['k']
    docs_k_shingles = []  # holds the k-shingles of each document

    for doc in document_list.values():
        doc_shingle = set()
        for i in range(len(doc) - k + 1):
            doc_shingle.add(doc[i:i + k])
        docs_k_shingles.append(list(doc_shingle))

    return list(docs_k_shingles)


# METHOD FOR TASK 2
# Creates a signatures set of the documents from the k-shingles list
def signature_set(k_shingles):
    '''
    :param k_shingles: list of k-shingles of each document
    :return: list of signature sets of each document (inverse index)
    '''
    docs_sig_sets = []

    shingle_array = np.array(k_shingles[0])

    # Create a union of all the shingles
    for k_shingle in k_shingles:
        shingle_array = np.union1d(shingle_array, np.array(k_shingle))
    
    # Create dict for speed optimization
    shingle_dict = {}
    for index, shingle in enumerate(shingle_array):
        shingle_dict[shingle] = index

    length = len(shingle_array)
    # Create the signature sets
    for docs_shingle in k_shingles:
        signature = list(np.zeros(length, dtype=int))
        for shingle in docs_shingle:
            index = shingle_dict[shingle]
            signature[index] = 1
        docs_sig_sets.append(list(signature))

    return docs_sig_sets


# Helper function for TASK 3
def make_a_hash_function(a, b, x, l):
    return (a * x + b) % l


def hash(a, b, x, N, p):
    return (a * x + b) %p % N

# METHOD FOR TASK 3
# Creates the minHash signatures after simulation of permutations
def minHash(docs_signature_sets):
    '''
    :param docs_signature_sets: list of signature sets of each document 
    :return: list of minHash signatures of each document
    '''
    min_hash_signatures = []
    
    permutations = parameters_dictionary['permutations']
    N = len(docs_signature_sets[0])

    min_hash_signatures = [[] for _ in range(len(docs_signature_sets))]

    for _ in range(permutations):
        a = random.randint(1, N)
        b = random.randint(1, N)
        p = randprime(N, 2*N)
        for signature_index, signature_set in enumerate(docs_signature_sets):
            min_hash = N
            for index, value in enumerate(signature_set):
                if value != 0:
                    min_hash = min(min_hash, hash(a, b, index, N, p))
            
            min_hash_signatures[signature_index] += min_hash,

    return min_hash_signatures


# METHOD FOR TASK 4
# Hashes the MinHash Signature Matrix into buckets and find candidate similar documents
def lsh(m_matrix):
    '''
    :param m_matrix: list of minHash signatures of each document
    :return: list of candidate sets of documents for checking similarity
    '''
    candidates = set()  # list of candidate sets of documents for checking similarity
    
    buckets = parameters_dictionary['buckets']
    r = parameters_dictionary['r']

    assert len(m_matrix[0]) % r == 0 # <-- the number of entries must be divisible by r

    m_matrix_banded = []

    # partition the signature matrix into bands
    for signature_index, signature in enumerate(m_matrix):
        banded_signature = []
        for i in range(0, len(signature), r):
            band = signature[i:i+r]
            banded_signature.append(band)
            
        m_matrix_banded.append(banded_signature)

    # one loop per band
    for band_index in range(len(m_matrix_banded[0])):   
        hash_buckets = {}
        # one loop per signature in the band
        for signature_index, signature in enumerate(m_matrix_banded):
            # simple hashing, taking the sum of the band and modding it by the number of buckets
            band = sum(signature[band_index])
            hash_value = band % buckets
            # add hash value into corresponding bucket
            if hash_value not in hash_buckets:
                hash_buckets[hash_value] = [signature_index]
            else:
                hash_buckets[hash_value].append(signature_index)

        # check for collisions
        for bucket in hash_buckets.values():
            if len(bucket) > 1: # <-- not consider buckets with one signature
                for i in range(len(bucket)):
                    for j in range(i + 1, len(bucket)):
                        candidate_pair = (bucket[i], bucket[j])
                        if candidate_pair not in candidates:
                            candidates.add(candidate_pair)
    return list(candidates)


# Helper function for TASK 5
def similarity(doc1, doc2):
    length = len(doc1)
    matches = 0
    for i in range(length):
        if doc1[i] == doc2[i]:
            matches += 1
    return matches / length


# METHOD FOR TASK 5
# Calculates the similarities of the candidate documents
def candidates_similarities(candidate_docs, min_hash_matrix):
    '''
    :param candidate_docs: list of candidate sets of documents for checking similarity
    :param min_hash_matrix: list of minHash signatures of each document
    :return: list of similarities of the candidate documents
    '''
    similarity_matrix = []

    for candidate in candidate_docs:
        doc1 = min_hash_matrix[candidate[0]]
        doc2 = min_hash_matrix[candidate[1]]
        sim = similarity(doc1, doc2)

        similarity_matrix.append(((candidate[0], candidate[1]), sim))

    return similarity_matrix


# Helper function for TASK 6
def getFileName(doc_index):
    doc_name = str(doc_index)
    for _ in range(3 - len(doc_name)):
        doc_name = '0' + doc_name
    return doc_name + '.txt'


# METHOD FOR TASK 6
# Returns the document pairs of over t% similarity
def return_results(lsh_similarity_matrix):
    document_pairs = set()
    t = parameters_dictionary['t']
    for row in lsh_similarity_matrix:
        if row[1] >= t:
            a = getFileName(row[0][0])
            b = getFileName(row[0][1])
            document_pairs.add((a, b))

    return list(document_pairs)


# METHOD FOR TASK 6
def count_false_neg_and_pos(lsh_similarity_matrix, naive_similarity_matrix):
    false_negatives = 0
    false_positives = 0
    positive = 0

    actual_positive = 0

    length = len(document_list)
    t = parameters_dictionary['t']

    print("There are " + str(len(lsh_similarity_matrix)) + " candidate pairs.")

    for candidate_pair in lsh_similarity_matrix:
        i = candidate_pair[0][0]
        j = candidate_pair[0][1]
        triangle_index = get_triangle_index(i, j, length)
        
        naive_similarity = naive_similarity_matrix[triangle_index]
        candidate_similarity = candidate_pair[1]

        if candidate_similarity >= t:
            positive += 1
        if candidate_similarity >= t and naive_similarity < t:
            false_positives += 1
        elif candidate_similarity < t and naive_similarity >= t:
            false_negatives += 1

    print("There were a total of " + str(positive) + " similarities above treshold value, out of which " + str(positive - false_positives) + " are true positive. \n")

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

