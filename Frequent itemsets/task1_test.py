# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 22:15:58 2021

@author: 13451
"""


from pyspark import SparkContext
import itertools
import time
import sys

args = sys.argv[:]

case = int(args[1])
support = int(args[2])
sc = SparkContext(master='local[*]',appName='AS2T1')
filepath = args[3]
start = time.clock()
raw = sc.textFile(filepath)
header = raw.first()
lines = raw.filter(lambda row:row != header)

#CASE 1
if case ==1:
    lines = lines.map(lambda r: (r.split(',')[0],r.split(',')[1]))
else: #CASE2
    lines = lines.map(lambda r: (r.split(',')[1],r.split(',')[0])).repartition(6)
    
def into_set(x):
    return set(x)

baskets = lines.groupByKey().mapValues(into_set)
total = baskets.count()

def hash_value(x): #add values in a set, x is a set
    value1 = 0
    for item in x:
        value1 += int(item)
    value2 = 1
    for item in x:
        value2 *= int(item)
    return (value1+value2)%500
        
def find_frequent_candidates(iterator): #A-Priori & PCY
    global support
    global total
    count_table = dict() # maintain an original hashtable to find frequent 1-item itemsets
    pre_hash = dict() #maintain an hashtable to pre-filter 2-item itemsets
    num_of_baskets = 0  # get number of baskets in this partition
    data = []
    for element in iterator:
        data.append(element)
        for candidates in itertools.combinations(element[1],2):
            if hash_value(candidates) not in pre_hash:
                pre_hash[hash_value(candidates)] = 1
            else:
                pre_hash[hash_value(candidates)] += 1
        for bid in element[1]:
            if bid not in count_table:
                count_table[bid]=1
            else:
                count_table[bid]+=1
        num_of_baskets +=1
    
    local_support = support*num_of_baskets/total
    one_item_candidates = sorted([key for key, value in count_table.items() if value > local_support])
    whole_candidates = one_item_candidates
    curlevel_candidates = set(one_item_candidates)
    level = 2
    while len(curlevel_candidates)>0:
        waiting_for_check = dict()
        
        if level ==2:
            for candidates in itertools.combinations(one_item_candidates,level):
                if hash_value(candidates) not in pre_hash:
                    flag = True
                    for checks in itertools.combinations(candidates,level-1):
                        if checks not in curlevel_candidates and checks[0] not in curlevel_candidates:
                            flag = False
                            break
                    if flag == True:
                        waiting_for_check[candidates] = 0
                elif pre_hash[hash_value(candidates)]>=support:
                    flag = True
                    for checks in itertools.combinations(candidates,level-1):
                        if checks not in curlevel_candidates and checks[0] not in curlevel_candidates:
                            flag = False
                            break
                    if flag == True:
                        waiting_for_check[candidates] = 0
        else:
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
        
        #del iterator
        #iterator = iter(data)
        for element in data:
            for key in waiting_for_check.keys():
                if set(key) <= element[1]:
                    waiting_for_check[key]+=1
        
        level+=1                
        curlevel_candidates = set(key for key, value in waiting_for_check.items() if value > local_support)
        whole_candidates = whole_candidates + list(curlevel_candidates)
    
    yield whole_candidates
    

candidates = baskets.mapPartitions(find_frequent_candidates).reduce(lambda a,b:list(set(a).union(set(b))))
t = time.clock()-start
print("Duration1:"+ str(t))

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

   
frequent = baskets.mapPartitions(count_candidates).reduceByKey(lambda a,b: a+b)\
                .filter(lambda r:r[1]>=support).map(lambda r:r[0]).collect()

t = time.clock()-start
print("Duration2:"+ str(t))

result_can = print_candidates(candidates)  
result_fre = print_candidates(frequent)


f = open(args[4],'w')
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
m = time.clock()-start
print("Duration:"+ str(m))
