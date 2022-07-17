# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 11:43:13 2021

@author: 13451
"""

from pyspark import SparkContext
import json

sc = SparkContext(master='local[*]',appName='AS1T1')
filepath = "jiahaosu@usc.edu_resource_1575654_s1575654_516245_Sep_18_2021_11-40-10am_PDT/resource/asnlib/publicdata/test_review.json"
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
file_name = 'output1.json'
with open(file_name,'w') as f:
    json.dump(ans, f, indent = 4)