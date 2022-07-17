# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 11:34:06 2021

@author: 13451
"""

from pyspark import SparkContext
import json
import time
import sys

args = sys.argv[:]

sc = SparkContext(master='local[*]',appName='AS1T2')
sc.setLogLevel("ERROR")
filepath = args[1]
n_partition = int(args[3])
raw = sc.textFile(filepath).map(lambda row:json.loads(row)).repartition(n_partition)

#default
n_partition_d = raw.getNumPartitions()

def count_item(iterator): 
    yield sum(1 for _ in iterator) 

n_items_d = raw.mapPartitions(count_item).collect() 

start_1 = time.clock()
top10_business = raw.map(lambda row:(row['business_id'],1)).reduceByKey(lambda a,b:a+b)\
                .sortBy(lambda row: (-row[1],row[0]),ascending =True).take(10)
t1 = time.clock()-start_1

print("exe time default is" + str(t1))

#customized
def hash_self(key):
    '''
    sum_ascii=0
    for items in key:
        sum_ascii+= ord(items)
    bucket_class = sum_ascii%12
    '''
    return hash(key)
        
paired = raw.map(lambda row:(row['business_id'],row)).partitionBy(n_partition,hash_self)
n_partition_c = paired.getNumPartitions()
n_items_c = paired.mapPartitions(count_item).collect()
start_2 = time.clock()
top10_business = paired.map(lambda row:(row[0],1)).reduceByKey(lambda a,b:a+b)\
                .sortBy(lambda row: (-row[1],row[0]),ascending =True).take(10)
t2 = time.clock()-start_2

print("exe time customized is" + str(t2))

ans = {"default":{"n_partition":n_partition_d,"n_items":n_items_d,"exe_time":t1},\
       "customized":{"n_partition":n_partition_c,"n_items":n_items_c,"exe_time":t2}}

file_name = args[2]
with open(file_name,'w') as f:
    json.dump(ans, f, indent = 4)