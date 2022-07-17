# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 13:23:45 2021

@author: 13451
"""


from pyspark import SparkContext
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import time
import sys
import os
from sklearn.metrics import mean_squared_error

args = sys.argv[:]

folder_path = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]
sc = SparkContext(master='local[*]',appName='AS3T22')

start = time.time()
train_rdd = sc.textFile(os.path.join(folder_path, 'yelp_train.csv'))
train_header = train_rdd.first()
train = train_rdd.filter(lambda x: x != train_header).map(lambda x: x.split(','))

test_rdd = sc.textFile(test_file)
test_header = test_rdd.first()
test = test_rdd.filter(lambda x: x != test_header).map(lambda x: x.split(','))

user_rdd = sc.textFile(os.path.join(folder_path, 'user.json')).map(json.loads)\
    .map(lambda x: (x["user_id"], (x["review_count"], x["useful"], x["fans"], x["average_stars"])))
user_map = user_rdd.collectAsMap()

business_rdd = sc.textFile(os.path.join(folder_path, 'business.json')).map(json.loads)\
    .map(lambda x: (x['business_id'], (x['stars'], x['review_count'])))
business_map = business_rdd.collectAsMap()

avg_users_rating = user_rdd.map(lambda x: x[1][3]).mean()
avg_users_reviewcount = user_rdd.map(lambda x: x[1][0]).mean()
avg_businesses_rating = business_rdd.map(lambda x: x[1][0]).mean()
avg_businesses_reviewcount = business_rdd.map(lambda x: x[1][1]).mean()

###processing feature
def generate_feature_train(row):
    user_id = row[0]
    business_id = row[1]
    if user_id in user_map:
        user_review_count = user_map[user_id][0]
        user_review_useful = user_map[user_id][1]
        user_fans = user_map[user_id][2]
        user_avg_rating= user_map[user_id][3]
    else:
        user_review_count= avg_users_reviewcount
        user_review_useful = 0
        user_fans = 0
        user_avg_rating= avg_users_rating
    if business_id in business_map:
        business_avg_rating = business_map[business_id][0]
        business_review_count = business_map[business_id][1]
    else:
        business_avg_rating = avg_businesses_rating 
        business_review_count = avg_businesses_reviewcount
    
    return [hash(row[0]),hash(row[1]),user_review_count,user_review_useful,user_fans,user_avg_rating,business_avg_rating,business_review_count,float(row[2])]

def generate_feature_test(row):
    user_id = row[0]
    business_id = row[1]
    if user_id in user_map:
        user_review_count = user_map[user_id][0]
        user_review_useful = user_map[user_id][1]
        user_fans = user_map[user_id][2]
        user_avg_rating= user_map[user_id][3]
    else:
        user_review_count= avg_users_reviewcount
        user_review_useful = 0
        user_fans = 0
        user_avg_rating= avg_users_rating
    if business_id in business_map:
        business_avg_rating = business_map[business_id][0]
        business_review_count = business_map[business_id][1]
    else:
        business_avg_rating = avg_businesses_rating 
        business_review_count = avg_businesses_reviewcount
    
    return [row[0],row[1],hash(row[0]),hash(row[1]),user_review_count,user_review_useful,user_fans,user_avg_rating,business_avg_rating,business_review_count]

train_df = pd.DataFrame(train.map(lambda row:generate_feature_train(row)).collect(),\
                        columns = ['userid_h','busid_h','user_review_count','user_review_useful','user_fans','user_avg_rating','business_avg_rating','business_review_count','real_rating'])
test_df = pd.DataFrame(test.map(lambda row:generate_feature_train(row)).collect(),\
                        columns = ['userid_h','busid_h','user_review_count','user_review_useful','user_fans','user_avg_rating','business_avg_rating','business_review_count','real_rating'])
#test_df = pd.DataFrame(test.map(lambda row:generate_feature_test(row)).collect(),\
#                       columns = ['user_id','bus_id','userid_h','busid_h','user_review_count','user_review_useful','user_fans','user_avg_rating','business_avg_rating','business_review_count'])
x_train = train_df.iloc[:,:-1]
y_train = train_df.iloc[:,-1]
x_test = test_df.iloc[:,:-1]
y_test = test_df.iloc[:,-1]
#x_test = test_df.iloc[:,2:]
#y_test = test_df.iloc[:,-1]

rec_xgb = xgb.XGBRegressor(learning_rate=0.3)
rec_xgb.fit(x_train, y_train)
y_train_pred = rec_xgb.predict(x_train)
y_test_pred = rec_xgb.predict(x_test)
print("train_RMSE : ", np.sqrt(mean_squared_error(y_train,y_train_pred)))
print("text_RMSE : ", np.sqrt(mean_squared_error(y_test,y_test_pred)))
#result = pd.concat([test_df.iloc[:,[0,1]],pd.DataFrame(y_test_pred)],axis=1)

#result.to_csv(output_file, header=['user_id', ' business_id', ' prediction'], index=False, sep=',', mode='w')

m = time.time()-start
print("Duration:"+ str(m))
