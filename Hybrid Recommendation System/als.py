# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 20:51:48 2021

@author: 13451
"""

from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StructField, StructType, FloatType,IntegerType
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import time
from datetime import datetime
import sys
import os
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from math import sqrt,floor
import random

args = sys.argv[:]

folder_path = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]
sc = SparkContext(master='local[*]',appName='competition')
spark = SparkSession(sc)

start = time.time()
train_rdd = sc.textFile(os.path.join(folder_path, 'yelp_train.csv'))
train_header = train_rdd.first()
train = train_rdd.filter(lambda x: x != train_header).map(lambda x: x.split(','))
train_user_map_1 = train.map(lambda row:row[0]).distinct().zipWithIndex().map(lambda row:(row[1],row[0])).collectAsMap()
train_user_map_2 = train.map(lambda row:row[0]).distinct().zipWithIndex().collectAsMap()
train_biz_map_1 = train.map(lambda row:row[1]).distinct().zipWithIndex().map(lambda row:(row[1],row[0])).collectAsMap()
train_biz_map_2 = train.map(lambda row:row[1]).distinct().zipWithIndex().collectAsMap()
train_data = train.map(lambda row:[train_user_map_2[row[0]],train_biz_map_2[row[1]],float(row[2])])

test_rdd = sc.textFile(test_file)
test_header = test_rdd.first()
test = test_rdd.filter(lambda x: x != test_header).map(lambda x: x.split(',')).filter(lambda row: row[0] in train_user_map_2 and row[1] in train_biz_map_2)
test_data = test.map(lambda row:[train_user_map_2[row[0]],train_biz_map_2[row[1]],float(row[2])])

distinct_user = train.map(lambda x: x[0]).distinct().collect()
distinct_business = train.map(lambda x:x[1]).distinct().collect()
raw_map = train.map(lambda x: ((x[0],x[1]), float(x[2]))).collectAsMap()
boost_set = []
for user in random.sample(distinct_user, round(len(distinct_user)*0.05)):
    for business in random.sample(distinct_business, round(len(distinct_business)*0.05)):
        if (user,business) not in raw_map:
            boost_set.append([user,business])
boost = sc.parallelize(boost_set)
boost_data = boost.map(lambda row:[train_user_map_2[row[0]],train_biz_map_2[row[1]]])

schema = StructType([
        StructField("user_id", IntegerType(), True),
        StructField("business_id", IntegerType(), True),
        StructField("stars", FloatType(), True)])
schema_b = StructType([
        StructField("user_id", IntegerType(), True),
        StructField("business_id", IntegerType(), True)])
train_df = spark.createDataFrame(train_data, schema=schema) 
test_df =  spark.createDataFrame(test_data, schema=schema) 
boost_df =  spark.createDataFrame(boost_data, schema=schema_b) 

#Create ALS model
als = ALS(userCol="user_id", itemCol="business_id", ratingCol="stars",coldStartStrategy="drop")

#Tune model using ParamGridBuilder
paramGrid = ParamGridBuilder()\
            .addGrid(als.regParam, [0.1, 0.01, 0.001])\
            .addGrid(als.maxIter, [3, 5, 10])\
            .addGrid(als.rank, [5, 10, 15])\
            .build()
# rank=15,maxiter=10,regparam = 0.1
# Define evaluator as RMSE
evaluator = RegressionEvaluator(metricName="rmse", labelCol="stars",
                                predictionCol="prediction")

# Build Cross validation 
crossval = CrossValidator(estimator=als,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)

#Fit ALS model to training data
model = als.fit(train_df)

#Extract best model from the tuning exercise using ParamGridBuilder
cvModel = crossval.fit(train_df)
best_model = cvModel.bestModel
print ("**Best Model**")
print (" Rank:"+str(best_model._java_obj.parent().getRank()))
print (" MaxIter:"+str(best_model._java_obj.parent().getMaxIter()))
print (" RegParam:"+str(best_model._java_obj.parent().getRegParam()))
predictions = best_model.transform(train_df)
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

#Generate predictions and evaluate using RMSE
predictions_test= best_model.transform(test_df)
def cutscore_v1(row):
    if float(row[3]) > 5:
        rating = 5.0
    elif float(row[3]) < 1:
        rating = 1.0
    else:
        rating = row[3]
    return [train_user_map_1[row[0]],train_biz_map_1[row[1]],rating]
test_predict_als = predictions_test.rdd.map(cutscore_v1).collect()
test_predict_als = pd.DataFrame(test_predict_als,columns =['user_id','business_id','prediction'])
test_data = pd.read_csv(test_file)
RMSE =  pd.merge(test_data,test_predict_als,on =['user_id','business_id'])
def mse(x):
    return (x['stars']-x['prediction'])**2
RMSE.loc[:,"mse"] = RMSE.apply(mse,axis=1)
print("RMSE : ", sqrt(RMSE['mse'].mean()))
#rmse = evaluator.evaluate(predictions_test)
#print("Root-mean-square error = " + str(rmse))

boost_prediction = best_model.transform(boost_df)

def cutscore_v2(row):
    if float(row[2]) > 5:
        rating = 5.0
    elif float(row[2]) < 1:
        rating = 1.0
    else:
        rating = row[2]
    return [train_user_map_1[row[0]],train_biz_map_1[row[1]],rating]

boost_rdd = boost_prediction.rdd.map(cutscore_v2)
whole_rdd = train.union(boost_rdd)

#user as key
user_map = whole_rdd.map(lambda x: (x[0], (x[1], float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
#business key
business_map = whole_rdd.map(lambda x: (x[1], (x[0], float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
#user avg rating
user_avg_map = whole_rdd.map(lambda x: (x[0], float(x[2]))).groupByKey().mapValues(lambda x: sum(x) / len(x)).collectAsMap()
#business avg rating
business_avg_map = whole_rdd.map(lambda x: (x[1], float(x[2]))).groupByKey().mapValues(lambda x: sum(x) / len(x)).collectAsMap() 
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
    if len(co_rated)==0 or 1: 
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
        sim =  hotness_sim*0.3 + category_sim*0.4 + rating_sim*0.3
    else:
        sim = -(hotness_sim*0.3 + category_sim*0.4 + rating_sim*0.3*(-1))
    
    return sim
    
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
                rating = 3.0
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
                    rating_denominator += abs(item[1])
                    rating_numerator += item[1]*user_map[user][item[0]]
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

test_predict_als = test.map(lambda row:compute_rating_weight(row,8)).map(cutscore).collect()
test_predict_als = pd.DataFrame(test_predict_als,columns =['user_id','business_id','prediction','weight'])
test_data = pd.read_csv(test_file)
RMSE =  pd.merge(test_data,test_predict_als,on =['user_id','business_id'])
def mse(x):
    return (x['stars']-x['prediction'])**2
RMSE.loc[:,"mse"] = RMSE.apply(mse,axis=1)
print("RMSE : ", sqrt(RMSE['mse'].mean()))