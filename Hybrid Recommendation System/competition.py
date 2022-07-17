# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:03:32 2021

@author: 13451
"""
############### Method description ##################
# Used XGboost to predict the ratings for (user,business) pairs
# Extracted some useful features from related dataset, such as the usage history length of the user
# Perform GridSearch and Cross-Validation to tune the parameters of XGBoost, such as n_estimators and max_depth and reg_alpha
# Why abandoned collaborative filtering? I found there are more than 20,000 distinct business_id and more than 10,000 distinct user_id,
# But there are only 450,000+ rows of (user,business,ratings), which means the rating matrix is extremely sparse.
# I have tried several ways to do content boosting (filling the huge rating matrix first and then do CF) by using XGBoost and ALS matrix decomposition
# Due to the running time limit, I can only randomly part of the user_id and business_id to generate the pseudo ratings, but I didn't get a good result.
# I also tried feature combination, using the ratings given by CF as input features of XGBoost, but also not perform well.
# In conclusion, CF really suffers from the data sparsity.

############### Error Distribution ##################
# >=0 and <1: 102241
# >=1 and <2: 32833
# >=2 and <3: 6146
# >=3 and <4: 822
# >=4: 2

#RMSE:0.9788993814858231

#Execution Time: 387.4818742275238 s

from pyspark import SparkContext
from math import sqrt
import json
import pandas as pd
import xgboost as xgb
#from sklearn.model_selection import GridSearchCV
import time
import sys
import os
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


############## model_based ############
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
    return (row['business_id'],(stars,review_count))

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
        bus_feat = list(business_info_map[business_id])
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
        bus_feat = list(business_info_map[business_id])
    else:
        bus_feat = avg_business_info
    features.extend(user_feat)
    features.extend(bus_feat)
    features.append(checkin_num)
    features.append(photo_num)
    return features


train_df = pd.DataFrame(train.map(lambda row:generate_feature_train(row)).collect(),columns = ['feat1','feat2','feat3','feat4','feat5','feat6','feat7','feat8','feat9','feat10','feat11','feat12','feat13','ratings'])
test_df = pd.DataFrame(test.map(lambda row:generate_feature_test(row)).collect(),columns = ['user_id','business_id','feat1','feat2','feat3','feat4','feat5','feat6','feat7','feat8','feat9','feat10','feat11','feat12','feat13',])

x_train = train_df.iloc[:,:-1]
y_train = train_df.iloc[:,-1]
x_test = test_df.iloc[:,2:]

'''
#grid search cv
parameters = {'reg_alpha':[10],
              'objective':['reg:squarederror'],
              'gamma': [0.6],
              'learning_rate': [.03, 0.05, .07], 
              'max_depth': [5,],
              'min_child_weight': [4],
              'subsample': [0.6,0.7,0.8,0.9],
              'colsample_bytree': [0.6,0.7,0.8,0.9],
              'n_estimators': [500]}
              
xgb_grid = GridSearchCV(xgb.XGBRegressor(),param_grid=parameters,cv = 5,n_jobs = 5,verbose=True)

xgb_grid.fit(x_train,y_train)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)
'''

rec_xgb = xgb.XGBRegressor(n_estimators = 1000,reg_alpha=10,gamma = 0.5,max_depth = 6,min_child_weight = 4,colsample_bytree = 0.8,subsample=0.9,learning_rate = 0.07,reg_lambda = 1)
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
test_predict_m.to_csv(output_file, header=['user_id', ' business_id', ' prediction'], index=False, sep=',', mode='w')
#RMSE
test_data = pd.read_csv(test_file)
RMSE =  pd.merge(test_data,test_predict_m,on =['user_id','business_id'])
def mse(x):
    return (x['stars']-x['prediction'])**2
def mae(x):
    return abs(x['stars']-x['prediction'])
RMSE.loc[:,"mse"] = RMSE.apply(mse,axis=1)
RMSE.loc[:,"mae"] = RMSE.apply(mae,axis=1)
print("RMSE : ", sqrt(RMSE['mse'].mean()))
print("ERROR DISTRIBUTION:")
print(">=0 and <1:" ,len(RMSE[(RMSE['mae']>=0) & (RMSE['mae']<1)]))
print(">=1 and <2:" ,len(RMSE[(RMSE['mae']>=1) & (RMSE['mae']<2)]))
print(">=2 and <3:" ,len(RMSE[(RMSE['mae']>=2) & (RMSE['mae']<3)]))
print(">=3 and <4:" ,len(RMSE[(RMSE['mae']>=3) & (RMSE['mae']<4)]))
print(">=4:" ,len(RMSE[RMSE['mae']>=4]))

m = time.time()-start
print("Duration:"+ str(m))


