# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 13:19:18 2021

@author: 13451
"""


from pyspark import SparkContext
import itertools

k = 30
support = 60
sc = SparkContext(master='local[*]',appName='AS2T2')
filepath = "resource/asnlib/publicdata/ta_feng_all_months_merged.csv"
raw = sc.textFile(filepath)

def process(row):
    if row[0].strip('"')== 'TRANSACTION_DT':
        return ('DATE-CUSTOMER_ID', 'PRODUCT_ID')
    else:
        return (row[0].strip('"').split('/')[0]+'/'+row[0].strip('"').split('/')[1]+'/'+row[0].strip('"').split('/')[2][-2:]+'-'+row[1].strip('"').lstrip('0'),int(row[5].strip('"').lstrip('0')))
    

#data preprocessing
processed = raw.map(lambda row:row.split(',')).map(process)
processed_out = processed.collect()
f = open('customer_product.csv','w')
for line in processed_out:
    f.write(','.join(str(d) for d in line))
    f.write('\n')
f.close()

#apply SON 
filepath = "customer_product.csv"
raw2 = sc.textFile(filepath)
header = raw2.first()
lines = raw2.filter(lambda row:row != header).map(lambda r:(r.split(',')[0],r.split(',')[1]))

def into_set(x):
    return set(x)
baskets = lines.groupByKey().mapValues(into_set).repartition(8)

baskets_filter = baskets.filter(lambda row: len(row[1])>k)
total = baskets_filter.count()


def find_frequent_candidates(iterator):
    global support
    global total
    count_table = dict() # maintain an original hashtable to find frequent 1-item itemsets
    num_of_baskets = 0  # get number of baskets in this partition
    data = []
    for element in iterator:
        data.append(element)
        for bid in element[1]:
            if bid not in count_table:
                count_table[bid]=1
            else:
                count_table[bid]+=1
        num_of_baskets +=1
    
    local_support = support*num_of_baskets/total
    one_item_candidates = set(key for key, value in count_table.items() if value > local_support)
    
    ans= [item for item in sorted(one_item_candidates)]
    curlevel_candidates = one_item_candidates
    level = 2
    while len(curlevel_candidates)>0:
        waiting_for_check = dict()
        
        for candidates in itertools.combinations(one_item_candidates,level):
            flag = True
            for checks in itertools.combinations(candidates,level-1):
                if checks not in curlevel_candidates and checks[0] not in curlevel_candidates:
                    flag = False
                    break
            if flag == True:
                waiting_for_check[candidates] = 0
        
        if len(waiting_for_check) ==0:
            break
        
        del iterator
        iterator = iter(data)
        for element in iterator:
            for key in waiting_for_check.keys():
                if set(key) <= element[1]:
                    waiting_for_check[key]+=1
        
        level+=1                
        curlevel_candidates = set(key for key, value in waiting_for_check.items() if value > local_support)
        for real_candidate in sorted(curlevel_candidates):
            ans.append(real_candidate)
    
    yield ans

def union_list(a,b):
    return list(set(a).union(set(b)))            

candidates = baskets_filter.mapPartitions(find_frequent_candidates).reduce(union_list)

def print_candidates(lst):
    max_len = 0
    for item in lst:
        if type(item) == str:
            max_len = max(max_len,1)
        else:
            max_len = max(max_len,len(item))
    whole = [[] for i in range(max_len)]
    for item in lst:
        if type(item) == str:
            whole[0].append("('"+item+"')")
        else:
            whole[len(item)-1].append(item)
    return whole

def count_candidates(iterator):
    global candidates
    count_table = dict()
    for element in iterator:
        for item in candidates:
            if type(item) == str:
                if set([item]) <= element[1]:
                    if item not in count_table:
                        count_table[item] =1
                    else:
                        count_table[item] +=1
            else:
                if set(item) <= element[1]:
                    if item not in count_table:
                        count_table[item] =1
                    else:
                        count_table[item] +=1
                
    return [(key,value) for key,value in count_table.items()]

result_can = print_candidates(candidates)    
frequent = baskets.mapPartitions(count_candidates).reduceByKey(lambda a,b: a+b)\
                .filter(lambda r:r[1]>=support).map(lambda r:r[0]).collect()
result_fre = print_candidates(frequent)

f = open('output2.txt','w')
f.write('Candidates:\n')
for subset in result_can:
    write = []
    for item in sorted(subset):
        write.append(str(item))
    f.write(','.join(write))
    f.write('\n')
    f.write('\n')
f.write('Frequent Itemsets:\n')
for subset in result_fre:
    write = []
    for item in sorted(subset):
        write.append(str(item))
    f.write(','.join(write))
    f.write('\n')
    f.write('\n')
f.close()
