# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 20:14:48 2021

@author: 13451
"""


from pyspark import SparkContext
from itertools import combinations
import time
import random
import sys

args = sys.argv[:]

filepath = args[1]
output_file = args[2]
sc = SparkContext(master='local[*]',appName='AS3T1')

start = time.time()
raw = sc.textFile(filepath)
header = raw.first()
lines = raw.filter(lambda row:row != header).map(lambda x: x.split(','))

users = lines.map(lambda x: x[0]).distinct().collect()
users.sort()
#hash the user_id str to value
user_map = {}
for i, u in enumerate(users):
    user_map[u] = i
num_users = len(user_map)

matrix = lines.map(lambda x: (x[1], user_map[x[0]])).groupByKey().mapValues(list).sortBy(lambda x: x[0])
char_matrix = matrix.collectAsMap()

def generate_signature(lst,a,b,m):
    hashvalues = []
    for val in lst:
        hval = (a*val + b) % m
        hashvalues.append(hval)
    return min(hashvalues)

random_hash_coef = [(random.randint(0,1000),random.randint(0,1000)) for i in range(60)]
sig_matrix = matrix.map(lambda x:(x[0],[generate_signature(x[1],a,b,num_users) for (a,b) in random_hash_coef]))

bands = 30
row_size = int(len(random_hash_coef) / bands)

def generate_pair(x): 
    return list(combinations(sorted(x),2))

def LSH(row, n_bands, n_rows):
    business_id = row[0]
    signature = row[1]
    signature_tuples = []
    for band in range(0,n_bands):
        re_signature = [band]
        re_signature.extend(signature[band*n_rows:(band*n_rows)+n_rows])
        signature_tuple = (tuple(re_signature), business_id)
        signature_tuples.append(signature_tuple)
    return signature_tuples

candidates = sig_matrix.flatMap(lambda row:LSH(row, bands, row_size)).groupByKey().mapValues(list)\
                .filter(lambda x: len(x[1]) > 1).flatMap(lambda x : generate_pair(x[1])).distinct()

def compute_js(row):
    c1 = set(char_matrix[row[0]])
    c2 = set(char_matrix[row[1]])
    jaccard_sim = len(c1 & c2) / len(c1 | c2)
    return (row[0],row[1], jaccard_sim)

result = candidates.map(lambda row: compute_js(row)).filter(lambda x: x[2] >= 0.5).sortBy(lambda x: (x[0], x[1])).collect()

with open(output_file, 'w') as f:
    f.write("business_id_1, business_id_2, similarity\n")
    for line in result:
        f.write(str(line[0]) + "," + str(line[1]) + "," + str(line[2]) + "\n")

m = time.time()-start
print("Duration:"+ str(m))
