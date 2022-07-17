# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 12:59:42 2021

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

args = sys.argv[:]

folder_path = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]
sc = SparkContext(master='local[*]',appName='AS3T23')
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

def compute_pearson_correlation(bid1,bid2):
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
        return 0
    average_1 = average_1/n
    average_2 = average_2/n
    
    for user in co_rated:
        coef_numerator += (bid1_users[user]-average_1)*(bid2_users[user]-average_2)
        coef_denominator_1 += (bid1_users[user]-average_1)**2
        coef_denominator_2 += (bid2_users[user]-average_2)**2
    if coef_denominator_1 == 0 or coef_denominator_2 == 0:
        return 0
    coef = coef_numerator/(sqrt(coef_denominator_1)*sqrt(coef_denominator_2))
    return coef
    
def rank(ele):
    return abs(ele[1])

def compute_rating_weight(row,n):
    user = row[0]
    business = row[1]
    if business not in business_map:
        if user not in user_map: #### new business & new user
            rating = 2.5
            weight = 0
        else: ### new business & old user
            rating = user_avg_map[user]
            weight = 0.05 #0.25,0.1
    else:
        if user not in user_map: ### old business & new user
            rating = business_avg_map[business]
            weight = 0.05
        else:  ### old business & old user
            neighbour = []
            for bus_neigh in user_map[user].keys():
                if compute_pearson_correlation(business,bus_neigh)> 0: #only consider pearson coef >0
                    neighbour.append((bus_neigh,compute_pearson_correlation(business,bus_neigh)))
            if len(neighbour) ==0:
                default_rating = (user_avg_map[user]+ business_avg_map[business])/2
                rating = default_rating
                weight = 0.1 #0.5,0.2
            else:
                neighbour.sort(key=rank,reverse = True)
                if len(neighbour) <=n:
                    selected_neighbour = neighbour
                else:
                    selected_neighbour = neighbour[:n]
                rating_denominator = 0
                rating_numerator = 0
                for item in selected_neighbour:
                    rating_denominator += abs(item[1])
                    rating_numerator += item[1]*user_map[user][item[0]]
                rating = rating_numerator/rating_denominator
                weight = selected_neighbour[-1] #0.75,0.8,1
    return [user,business,rating, weight]

def cutscore(row):
    if float(row[2]) > 5:
        row[2] = '5.0'
    elif float(row[2]) < 1:
        row[2] = '1.0'
    else:
        row[2] = row[2]
    return [row[0],row[1],row[2],row[3]]

test_predict_cf = test.map(lambda row:compute_rating_weight(row,8)).map(cutscore).collect()

############## model_based ############
user_rdd = sc.textFile(os.path.join(folder_path, 'user.json')).map(json.loads)\
    .map(lambda x: (x["user_id"], (x["review_count"], x["useful"], x["fans"], x["average_stars"])))
user_data_map = user_rdd.collectAsMap()

business_rdd = sc.textFile(os.path.join(folder_path, 'business.json')).map(json.loads)\
    .map(lambda x: (x['business_id'], (x['stars'], x['review_count'])))
business_data_map = business_rdd.collectAsMap()

avg_users_rating = user_rdd.map(lambda x: x[1][3]).mean()
avg_users_reviewcount = user_rdd.map(lambda x: x[1][0]).mean()
avg_businesses_rating = business_rdd.map(lambda x: x[1][0]).mean()
avg_businesses_reviewcount = business_rdd.map(lambda x: x[1][1]).mean()

###processing feature
def generate_feature_train(row):
    user_id = row[0]
    business_id = row[1]
    if user_id in user_data_map:
        user_review_count = user_data_map[user_id][0]
        user_review_useful = user_data_map[user_id][1]
        user_fans = user_data_map[user_id][2]
        user_avg_rating= user_data_map[user_id][3]
    else:
        user_review_count= avg_users_reviewcount
        user_review_useful = 0
        user_fans = 0
        user_avg_rating= avg_users_rating
    if business_id in business_data_map:
        business_avg_rating = business_data_map[business_id][0]
        business_review_count = business_data_map[business_id][1]
    else:
        business_avg_rating = avg_businesses_rating 
        business_review_count = avg_businesses_reviewcount
    
    return [hash(row[0]),hash(row[1]),user_review_count,user_review_useful,user_fans,user_avg_rating,business_avg_rating,business_review_count,float(row[2])]

def generate_feature_test(row):
    user_id = row[0]
    business_id = row[1]
    if user_id in user_data_map:
        user_review_count = user_data_map[user_id][0]
        user_review_useful = user_data_map[user_id][1]
        user_fans = user_data_map[user_id][2]
        user_avg_rating= user_data_map[user_id][3]
    else:
        user_review_count= avg_users_reviewcount
        user_review_useful = 0
        user_fans = 0
        user_avg_rating= avg_users_rating
    if business_id in business_data_map:
        business_avg_rating = business_data_map[business_id][0]
        business_review_count = business_data_map[business_id][1]
    else:
        business_avg_rating = avg_businesses_rating 
        business_review_count = avg_businesses_reviewcount
    
    return [row[0],row[1],hash(row[0]),hash(row[1]),user_review_count,user_review_useful,user_fans,user_avg_rating,business_avg_rating,business_review_count]


train_df = pd.DataFrame(train.map(lambda row:generate_feature_train(row)).collect(),\
                        columns = ['userid_h','busid_h','user_review_count','user_review_useful','user_fans','user_avg_rating','business_avg_rating','business_review_count','real_rating'])
test_df = pd.DataFrame(test.map(lambda row:generate_feature_test(row)).collect(),\
                       columns = ['user_id','bus_id','userid_h','busid_h','user_review_count','user_review_useful','user_fans','user_avg_rating','business_avg_rating','business_review_count'])
x_train = train_df.iloc[:,:-1]
y_train = train_df.iloc[:,-1]
x_test = test_df.iloc[:,2:]

rec_xgb = xgb.XGBRegressor(gamma=0.6,objective='reg:linear', reg_alpha=10)
rec_xgb.fit(x_train, y_train)
y_train_pred = rec_xgb.predict(x_train)
y_test_pred = rec_xgb.predict(x_test)

test_predict_m = pd.concat([test_df.iloc[:,[0,1]],pd.DataFrame(y_test_pred)],axis=1)
test_predict_m.columns=['user_id','business_id','prediction']

### hybrid recommendation ###
test_predict_cf = pd.DataFrame(test_predict_cf,columns =['user_id','business_id','prediction','weight'])
prediction = pd.merge(test_predict_cf,test_predict_m,on =['user_id','business_id'])

def final_prediction(x):
    return x['weight']*x['prediction_x']+(1-x['weight'])*x['prediction_y']
prediction.loc[:,"prediction"] = prediction.apply(final_prediction,axis=1)
result = prediction[['user_id','business_id','prediction']]
result.to_csv(output_file, header=['user_id', ' business_id', ' prediction'], index=False, sep=',', mode='w')
#RMSE

test_data = pd.read_csv(test_file)
RMSE =  pd.merge(test_data,result,on =['user_id','business_id'])
def mse(x):
    return (x['stars']-x['prediction'])**2
RMSE.loc[:,"mse"] = RMSE.apply(mse,axis=1)
print("RMSE : ", sqrt(RMSE['mse'].mean()))


m = time.time()-start
print("Duration:"+ str(m))
