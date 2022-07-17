# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 14:27:25 2021

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
sc.setLogLevel("ERROR")

start = time.time()
train_rdd = sc.textFile(os.path.join(folder_path, 'yelp_train.csv'))
train_header = train_rdd.first()
train = train_rdd.filter(lambda x: x != train_header).map(lambda x: x.split(','))

test_rdd = sc.textFile(test_file)
test_header = test_rdd.first()
test = test_rdd.filter(lambda x: x != test_header).map(lambda x: x.split(','))


############## model  ######################
def extract_user_feature(row):
    review_count = row['review_count']
    date = row['yelping_since'].split('-')
    history_len = (datetime.now() - datetime(int(date[0]),int(date[1]),int(date[2]))).days
    friends_num = len(row['friends'].split(',')) if row['friends'] != 'None' else 0
    useful = row['useful']
    funny = row['funny']
    cool = row['cool']
    fans = row['fans']
    elite = len(row['elite'].split(',')) if row['elite'] != 'None' else 0
    average_stars = row['average_stars']
    compliment_hot = row['compliment_hot']
    compliment_more = row['compliment_more']
    compliment_profile = row['compliment_profile']
    compliment_cute = row['compliment_cute']
    compliment_list = row['compliment_list']
    compliment_note = row['compliment_note']
    compliment_plain = row['compliment_plain']
    cpmpliment_cool = row['compliment_cool']
    compliment_funny = row['compliment_funny']
    compliment_writer = row['compliment_writer']
    compliment_photos = row['compliment_photos']
    return (row['user_id'],(review_count,history_len,friends_num,useful,funny,cool,fans,elite,average_stars,compliment_hot,\
                            compliment_more,compliment_profile,compliment_cute,compliment_list,compliment_note,compliment_plain,\
                            cpmpliment_cool,compliment_funny,compliment_writer,compliment_photos))

def extract_business_feature(row):
    stars = row['stars']
    review_count = row['review_count']
    category = row['categories'].split(',') if row['categories'] is not None else []
    return (row['business_id'],(stars,review_count,category))

def extract_checkin_feature(row):
    bus_id = row['business_id']
    check_sum = 0
    for key in row['time'].keys():
        check_sum += row['time'][key]
    return (bus_id,check_sum)

def process_user_feature(row):
    user_id = row[0]
    user_info = row[1]
    vote_sent = user_info[3]+user_info[4]+user_info[5]
    compliment_received =0 
    for i in range(9,20):
        compliment_received += user_info[i]
    return (user_id,(user_info[0],user_info[1],user_info[2],vote_sent,user_info[7],user_info[8],compliment_received))

checkin = sc.textFile(os.path.join(folder_path, 'checkin.json')).map(json.loads).map(extract_checkin_feature)
checkin_map = checkin.collectAsMap()
avg_checkin = checkin.map(lambda x:x[1]).mean() 

photo = sc.textFile(os.path.join(folder_path, 'photo.json')).map(json.loads).map(lambda row:(row['business_id'],row['photo_id']))\
        .groupByKey().map(lambda row:(row[0],len(row[1])))
photo_map = photo.collectAsMap()
avg_photo = photo.map(lambda x:x[1]).mean() 

raw_user_info = sc.textFile(os.path.join(folder_path, 'user.json')).map(json.loads).map(extract_user_feature)
re_user_info = raw_user_info.map(process_user_feature)
re_user_info_map = re_user_info.collectAsMap()
avg_user_info = [re_user_info.map(lambda x:x[1][i]).mean() for i in range(7)]

business_info = sc.textFile(os.path.join(folder_path, 'business.json')).map(json.loads).map(extract_business_feature)
business_info_map = business_info.collectAsMap()
avg_business_info = [business_info.map(lambda x:x[1][i]).mean() for i in range(2)]

def generate_feature_train(row):
    user_id = row[0]
    business_id = row[1]
    features = [hash(row[0]),hash(row[1])]
    if business_id in checkin_map:
        checkin_num = checkin_map[business_id]
    else:
        checkin_num = avg_checkin
    if business_id in photo_map:
        photo_num = photo_map[business_id]
    else:
        photo_num = avg_photo
    if user_id in re_user_info_map:
        user_feat = list(re_user_info_map[user_id])
    else:
        user_feat = avg_user_info
    if business_id in business_info_map:
        bus_feat = list(business_info_map[business_id])[:2]
    else:
        bus_feat = avg_business_info
    features.extend(user_feat)
    features.extend(bus_feat)
    features.append(checkin_num)
    features.append(photo_num)
    features.append(float(row[2]))
    return features

def generate_feature_test(row):
    user_id = row[0]
    business_id = row[1]
    features = [user_id,business_id,hash(row[0]),hash(row[1])]
    if business_id in checkin_map:
        checkin_num = checkin_map[business_id]
    else:
        checkin_num = avg_checkin
    if business_id in photo_map:
        photo_num = photo_map[business_id]
    else:
        photo_num = avg_photo
    if user_id in re_user_info_map:
        user_feat = list(re_user_info_map[user_id])
    else:
        user_feat = avg_user_info
    if business_id in business_info_map:
        bus_feat = list(business_info_map[business_id])[:2]
    else:
        bus_feat = avg_business_info
    features.extend(user_feat)
    features.extend(bus_feat)
    features.append(checkin_num)
    features.append(photo_num)
    return features


train_df = pd.DataFrame(train.map(lambda row:generate_feature_train(row)).collect(),columns = ['feat1','feat2','feat3','feat4','feat5','feat6','feat7','feat8','feat9','feat10','feat11','feat12','feat13','ratings'])
test_df = pd.DataFrame(test.map(lambda row:generate_feature_test(row)).collect(),columns = ['user_id','business_id','feat1','feat2','feat3','feat4','feat5','feat6','feat7','feat8','feat9','feat10','feat11','feat12','feat13'])

x_train = train_df.iloc[:,:-1]
y_train = train_df.iloc[:,-1]
x_test = test_df.iloc[:,2:]

rec_xgb = xgb.XGBRegressor(objective="reg:linear",reg_alpha=10,gamma = 0.6,max_depth = 5,min_child_weight = 4,n_estimators = 500,colsample_bytree = 0.9,subsample=0.9,learning_rate = 0.07)
rec_xgb.fit(x_train, y_train)
y_test_pred = rec_xgb.predict(x_test)

test_predict_m = pd.concat([test_df.iloc[:,[0,1]],pd.DataFrame(y_test_pred)],axis=1)
test_predict_m.columns=['user_id','business_id','prediction_1']
def cutscore_df(x):
    if x['prediction_1']>5:
        return 5.0
    elif x['prediction_1']<1:
        return 1.0
    else:
        return x['prediction_1']
test_predict_m.loc[:,'prediction'] = test_predict_m.apply(cutscore_df,axis = 1)
test_predict_m = test_predict_m.drop(labels = ['prediction_1'],axis = 1)
#test_predict_m.to_csv(output_file, header=['user_id', ' business_id', ' prediction'], index=False, sep=',', mode='w')

############## content-boosted als ##############
train_user_map_1 = train.map(lambda row:row[0]).distinct().zipWithIndex().map(lambda row:(row[1],row[0])).collectAsMap()
train_user_map_2 = train.map(lambda row:row[0]).distinct().zipWithIndex().collectAsMap()
train_biz_map_1 = train.map(lambda row:row[1]).distinct().zipWithIndex().map(lambda row:(row[1],row[0])).collectAsMap()
train_biz_map_2 = train.map(lambda row:row[1]).distinct().zipWithIndex().collectAsMap()
train_data = train.map(lambda row:[train_user_map_2[row[0]],train_biz_map_2[row[1]],float(row[2])])

test_f = test.filter(lambda row: row[0] in train_user_map_2 and row[1] in train_biz_map_2)
test_data = test_f.map(lambda row:[train_user_map_2[row[0]],train_biz_map_2[row[1]],float(row[2])])

distinct_user = train.map(lambda x: x[0]).distinct().collect()
distinct_business = train.map(lambda x:x[1]).distinct().collect()
raw_map = train.map(lambda x: ((x[0],x[1]), float(x[2]))).collectAsMap()
boost_set = []
for user in random.sample(distinct_user, round(len(distinct_user)*0.1)):
    for business in random.sample(distinct_business, round(len(distinct_business)*0.1)):
        if (user,business) not in raw_map:
            boost_set.append([user,business])
boost_data = sc.parallelize(boost_set)

boost_df = pd.DataFrame(boost_data.map(lambda row:generate_feature_test(row)).collect(),columns = ['user_id','business_id','feat1','feat2','feat3','feat4','feat5','feat6','feat7','feat8','feat9','feat10','feat11','feat12','feat13'])
x_boost = boost_df.iloc[:,2:]
y_boost_pred = rec_xgb.predict(x_boost)
boost_res = pd.concat([boost_df.iloc[:,[0,1]],pd.DataFrame(y_boost_pred)],axis=1)
boost_res_rdd = spark.createDataFrame(boost_res).rdd.map(lambda row:[train_user_map_2[row[0]],train_biz_map_2[row[1]],float(row[2])])

whole_rdd = train_data.union(boost_res_rdd)

schema = StructType([
        StructField("user_id", IntegerType(), True),
        StructField("business_id", IntegerType(), True),
        StructField("stars", FloatType(), True)])
train_df = spark.createDataFrame(whole_rdd, schema=schema)
schema_b = StructType([
        StructField("user_id", IntegerType(), True),
        StructField("business_id", IntegerType(), True)])
test_df =  spark.createDataFrame(test_data, schema=schema)  
#Create ALS model
als = ALS(rank = 15, maxIter=10,regParam=0.1,userCol="user_id", itemCol="business_id", ratingCol="stars",coldStartStrategy="drop")
# Define evaluator as RMSE
evaluator = RegressionEvaluator(metricName="rmse", labelCol="stars",
                                predictionCol="prediction")
model = als.fit(train_df)
prediction_train = model.transform(train_df)
rmse = evaluator.evaluate(prediction_train)
print("Root-mean-square error = " + str(rmse))

#Generate predictions and evaluate using RMSE
predictions_test= model.transform(test_df)
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



m = time.time()-start
print("Duration:"+ str(m))

