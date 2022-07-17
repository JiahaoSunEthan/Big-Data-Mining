# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:28:21 2021

@author: 13451
"""



from pyspark import SparkContext
from math import sqrt
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import time
import sys
import os
from sklearn.metrics import mean_squared_error
from datetime import datetime

args = sys.argv[:]

folder_path = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]
sc = SparkContext(master='local[*]',appName='competition')
sc.setLogLevel("ERROR")

start = time.time()

train_rdd = sc.textFile(os.path.join(folder_path, 'yelp_train.csv'))
train_header = train_rdd.first()
train = train_rdd.filter(lambda x: x != train_header).map(lambda x: x.split(','))

test_rdd = sc.textFile(test_file)
test_header = test_rdd.first()
test = test_rdd.filter(lambda x: x != test_header).map(lambda x: x.split(','))

############## item_based CF ##############

#user as key
user_map = train.map(lambda x: (x[0], (x[1], float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
#business key
business_map = train.map(lambda x: (x[1], (x[0], float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
#user avg rating
user_avg_map = train.map(lambda x: (x[0], float(x[2]))).groupByKey().mapValues(lambda x: sum(x) / len(x)).collectAsMap()
#business avg rating
business_avg_map = train.map(lambda x: (x[1], float(x[2]))).groupByKey().mapValues(lambda x: sum(x) / len(x)).collectAsMap() 

def extract_business_feature(row):
    stars = row['stars']
    review_count = row['review_count']
    category = row['categories'].split(',') if row['categories'] is not None else []
    return (row['business_id'],(stars,review_count,category))


business_info = sc.textFile(os.path.join(folder_path, 'business.json')).map(json.loads).map(extract_business_feature)
business_info_map = business_info.collectAsMap()


def compute_similarity(bid1,bid2):
    bid1_info = business_info_map[bid1]
    bid2_info = business_info_map[bid2]
    
    hotness_sim = (bid1_info[0]*bid2_info[0] + bid1_info[1]*bid2_info[1])/ (sqrt(bid1_info[0]**2+ bid1_info[1]**2)*sqrt(bid2_info[0]**2+ bid2_info[1]**2))
    if len(bid1_info[2]) == 0 or len(bid2_info[2])==0:
        category_sim = 0
    else:
        category_sim = len(set(bid1_info[2]).intersection(set(bid2_info[2])))/len(set(bid1_info[2]).union(set(bid2_info[2])))
        
    bid1_users = business_map[bid1]
    bid2_users = business_map[bid2]
    coef_denominator_1 = 0
    coef_denominator_2 = 0
    coef_numerator = 0
    average_1 = 0
    average_2 = 0
    n = 0
    co_rated = []
    for user in bid1_users.keys():
        if user in bid2_users.keys():
            average_1 += bid1_users[user]
            average_2 += bid2_users[user]
            n+=1
            co_rated.append(user)
    if len(co_rated)==0 or 1: ##only consider co-rated user >=10
        rating_sim = 0
    else:
        average_1 = average_1/n
        average_2 = average_2/n
    
    for user in co_rated:
        coef_numerator += (bid1_users[user]-average_1)*(bid2_users[user]-average_2)
        coef_denominator_1 += (bid1_users[user]-average_1)**2
        coef_denominator_2 += (bid2_users[user]-average_2)**2
    if coef_denominator_1 == 0 or coef_denominator_2 == 0:
        rating_sim = 0
    else:
        rating_sim = coef_numerator/(sqrt(coef_denominator_1)*sqrt(coef_denominator_2))
    
    if rating_sim>0:
        sim =  hotness_sim*0.2 + category_sim*0.3 + rating_sim*0.5
    else:
        sim = -(hotness_sim*0.2 + category_sim*0.3 + rating_sim*0.5*(-1))
    
    return rating_sim
    
def rank(ele):
    return abs(ele[1])

def compute_rating_weight(row,n):
    user = row[0]
    business = row[1]
    if business not in business_map:
        if user not in user_map: #### new business & new user
            rating = 3.0
            weight = 0
        else: ### new business & old user
            rating = user_avg_map[user]
            weight = 0
    else:  ### old business & old user
        if user not in user_map: ### old business & new user
            rating = business_avg_map[business]
            weight = 0
        else:
            neighbour = []
            for bus_neigh in user_map[user].keys():
                if compute_similarity(business,bus_neigh)> 0: #only consider pearson coef >0
                    neighbour.append((bus_neigh,compute_similarity(business,bus_neigh)))
            if len(neighbour) ==0:
                rating = (user_avg_map[user]+ business_avg_map[business])/2
                weight = 0 #0.5,0.2
            else:
                neighbour.sort(key=rank,reverse = True)
                if len(neighbour) <=n:
                    selected_neighbour = neighbour
                else:
                    selected_neighbour = neighbour[:n]
                rating_denominator = 0
                rating_numerator = 0
                for item in selected_neighbour:
                    rating_denominator += abs(item[1]**2)
                    rating_numerator += (item[1]**2)*user_map[user][item[0]]
                rating = rating_numerator/rating_denominator
                weight = selected_neighbour[-1][1] #0.75,0.8,1
    return [user,business,rating, weight]

def cutscore(row):
    if float(row[2]) > 5:
        row[2] = 5.0
    elif float(row[2]) < 1:
        row[2] = 1.0
    else:
        row[2] = row[2]
    return [row[0],row[1],row[2],row[3]]

test_predict_cf = test.map(lambda row:compute_rating_weight(row,6)).map(cutscore).collect()
test_predict_cf = pd.DataFrame(test_predict_cf,columns =['user_id','business_id','prediction','weight'])
test_data = pd.read_csv(test_file)
RMSE =  pd.merge(test_data,test_predict_cf,on =['user_id','business_id'])
def mse(x):
    return (x['stars']-x['prediction'])**2
RMSE.loc[:,"mse"] = RMSE.apply(mse,axis=1)
print("RMSE : ", sqrt(RMSE['mse'].mean()))
