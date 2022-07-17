# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 21:42:44 2021

@author: 13451
"""


from pyspark import SparkContext
import json
import time
import sys

args = sys.argv[:]

sc = SparkContext(master='local[*]',appName='AS1T3')
filepath1 = args[1]
filepath2 = args[2]

review_raw = sc.textFile(filepath1).map(lambda row:json.loads(row))
business_raw = sc.textFile(filepath2).map(lambda row:json.loads(row))

#A
review = review_raw.map(lambda r:(r['business_id'],r['stars']))
business = business_raw.map(lambda r:(r['business_id'],r['city']))
r_b = business.join(review).map(lambda r:(r[1][0],(r[1][1],1)))\
        .reduceByKey(lambda a,b:(a[0]+b[0],a[1]+b[1]))\
        .map(lambda r:(r[0],r[1][0]/r[1][1]))\
        .sortBy(lambda row:(-row[1],row[0]),ascending =True).collect()

f = open(args[3],'w')
f.write('city,stars\n')
for items in r_b:
    f.writelines([items[0],',',str(items[1]),'\n'])
f.close()


#B
#use python sort
data = business.join(review).map(lambda r:(r[1][0],(r[1][1],1)))\
        .reduceByKey(lambda a,b:(a[0]+b[0],a[1]+b[1]))\
        .map(lambda r:(r[0],r[1][0]/r[1][1])).collect()
start_1 = time.clock()
res1 = sorted(data,key = lambda x:(-x[1],x[0]))[:10]
for item in res1:
    print(item)
m1 = time.clock()-start_1
print("sort by python:"+ str(m1)+'s')

#use pyspark rdd sort
start_2 = time.clock()
data_rdd = sc.parallelize(data)
res2 = data_rdd.sortBy(lambda row:(-row[1],row[0]),ascending =True).take(10)
for item in res2:
    print(item)       
m2 = time.clock()-start_2
print("sort by spark:"+ str(m2)+'s')

ans = {"m1":m1,"m2":m2,"reason":"When dealing with small data, just like handling the subset of the data in my local environment,\
       python is faster than spark. But when it comes to big data,spark has better efficiency"}
file_name = args[4]
with open(file_name,'w') as f:
    json.dump(ans, f, indent = 4)

