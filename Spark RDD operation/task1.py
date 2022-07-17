# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 19:34:30 2021

@author: 13451
"""


from pyspark import SparkContext
import json
import sys

args = sys.argv[:]

sc = SparkContext(master='local[*]',appName='AS1T1')
filepath = args[1]
raw = sc.textFile(filepath).map(lambda row:json.loads(row))

#A
n_review = raw.count()

#B
n_review_2018 = raw.filter(lambda row: row['date'][:4] == '2018').count()

#C
n_user = raw.map(lambda row:row['user_id']).distinct().count()

#D
top10_user = raw.map(lambda row:(row['user_id'],1)).reduceByKey(lambda a,b:a+b)\
                .sortBy(lambda row: (-row[1],row[0]),ascending =True).take(10)
                
#E
n_business = raw.map(lambda row:row['business_id']).distinct().count()

#F
top10_business = raw.map(lambda row:(row['business_id'],1)).reduceByKey(lambda a,b:a+b)\
                .sortBy(lambda row: (-row[1],row[0]),ascending =True).take(10)
                
ans = {}
ans["n_review"]=n_review
ans["n_review_2018"]=n_review_2018
ans["n_user"]=n_user
ans["top10_user"]=top10_user
ans["n_business"]=n_business
ans["top10_business"]=top10_business
file_name = args[2]
with open(file_name,'w') as f:
    json.dump(ans, f, indent = 4)