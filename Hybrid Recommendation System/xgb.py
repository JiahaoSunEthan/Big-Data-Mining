# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 20:02:01 2021

@author: 13451
"""


from pyspark import SparkContext
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
from math import sqrt
from itertools import combinations

args = sys.argv[:]

folder_path = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]
sc = SparkContext(master='local[*]',appName='competition')

start = time.time()
train_rdd = sc.textFile(os.path.join(folder_path, 'yelp_train.csv'))
train_header = train_rdd.first()
train = train_rdd.filter(lambda x: x != train_header).map(lambda x: x.split(','))

test_rdd = sc.textFile(test_file)
test_header = test_rdd.first()
test = test_rdd.filter(lambda x: x != test_header).map(lambda x: x.split(','))

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

'''
reg_alpha_lst = [10 ** i for i in range(-4, 5)]
reg_lambda_lst=[0.05, 0.1, 1, 2, 3]
gamma_lst = [0.4,0.5,0.6,0.7,0.8]
max_depth_lst = [3,4,5,6,7]
min_child_weight_lst = [2,3,4,5,6]
n_estimators_lst = [425,450,475,500,525,550,575]
colsample_bytree_lst = [0.6,0.7,0.8,0.9]
subsample_lst = [0.6,0.7,0.8,0.9]
learning_rate_lst =[0.01, 0.05, 0.07, 0.1, 0.15]

test_data = pd.read_csv(test_file)
best_rmse =2
best_param = []
for param1 in n_estimators_lst:
    for param2 in max_depth_lst:
        for param3 in min_child_weight_lst:
#            for param4 in gamma_lst:
#                for param5 in colsample_bytree_lst:
#                    for param6 in subsample_lst:
#                        for param7 in reg_alpha_lst:
#                            for param8 in reg_lambda_lst:
#                                for param9 in learning_rate_lst:
                    rec_xgb = xgb.XGBRegressor(n_estimators=param1,max_depth=param2,min_child_weight=param3)
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
                    RMSE =  pd.merge(test_data,test_predict_m,on =['user_id','business_id'])
                    def mse(x):
                        return (x['stars']-x['prediction'])**2
                    RMSE.loc[:,"mse"] = RMSE.apply(mse,axis=1)
                    if sqrt(RMSE['mse'].mean())<=best_rmse:
                        best_rmse = sqrt(RMSE['mse'].mean())
                        best_param = [param1,param2,param3]
'''
rec_xgb = xgb.XGBRegressor(objective="reg:linear",reg_alpha=10,gamma = 0.6,max_depth = 5,min_child_weight = 4,n_estimators = 1000,colsample_bytree = 0.9,subsample=0.9,learning_rate = 0.07)
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
#RMSE
test_data = pd.read_csv(test_file)
RMSE =  pd.merge(test_data,test_predict_m,on =['user_id','business_id'])
def mse(x):
    return (x['stars']-x['prediction'])**2
RMSE.loc[:,"mse"] = RMSE.apply(mse,axis=1)
print("RMSE : ", sqrt(RMSE['mse'].mean()))
'''
#test_predict_m.to_csv(output_file, header=['user_id', ' business_id', ' prediction'], index=False, sep=',', mode='w')

# perform train_test validation to find best param

avg_test_rmse = []
alphas = [10 ** i for i in range(-10, 11)]
for alpha in alphas:
    xgb_reg= xgb.XGBRegressor(objective="reg:squarederror", reg_alpha=alpha)
    xgb_reg.fit(x_train, y_train)
    y_test_pred = xgb_reg.predict(x_test)
    avg_test_rmse.append((alpha,np.sqrt(mean_squared_error(y_test,y_test_pred))))
avg_test_rmse.sort(key=lambda x: x[1])
print("When alpha = {}, the validation mse reaches minimum of {}.".format(avg_test_rmse[0][0],
                                                               avg_test_rmse[0][1]))

rec_xgb = xgb.XGBRegressor(objective="reg:squarederror",reg_alpha=100)
rec_xgb.fit(x_train, y_train)
y_train_pred = rec_xgb.predict(x_train)
y_test_pred = rec_xgb.predict(x_test)
print("train_RMSE : ", np.sqrt(mean_squared_error(y_train,y_train_pred)))
print("test_RMSE : ", np.sqrt(mean_squared_error(y_test,y_test_pred)))

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
              
#{'gamma': 0.6, 'max_depth': 5, 'min_child_weight': 4, 'n_estimators': 500, 'objective': 'reg:linear', 'reg_alpha': 10}

xgb_grid = GridSearchCV(xgb.XGBRegressor(),param_grid=parameters,cv = 5,n_jobs = 5,verbose=True)

xgb_grid.fit(x_train,y_train)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

reg_alpha_lst = [10 ** i for i in range(-4, 5)]
reg_lambda_lst=[1.1,1.3,1.5,1.7,1.9]
gamma_lst = [0.4,0.5,0.6,0.7,0.8,0.9]
max_depth_lst = [3,4,5,6,7,8,9,10,11,12]
min_child_weight_lst = [2,3,4,5,6,7,8]
colsample_bytree_lst = [0.6,0.7,0.8,0.9]
subsample_lst = [0.95]
learning_rate_lst =[0.08,0.09]

n_estimators_list = [600,700,800,900,1000,1200,1400,1600,1800,3000]
test_data = pd.read_csv(test_file)
for param in n_estimators_list :
    #rec_xgb = xgb.XGBRegressor(n_estimators = 1000,max_depth =3,min_child_weight = 5,gamma = 0.4,colsample_bytree=0.9,subsample = 0.95,reg_alpha = 10,reg_lambda = 2,learning_rate=0.15)
    rec_xgb = xgb.XGBRegressor(n_estimators = param)
    #objective="reg:linear",reg_alpha=10,gamma = 0.6,max_depth = 5,min_child_weight = 4,n_estimators = 1000,colsample_bytree = 0.9,subsample=0.9,learning_rate = 0.07
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
    
    
    #RMSE
    #test_data = pd.read_csv(test_file)
    RMSE =  pd.merge(test_data,test_predict_m,on =['user_id','business_id'])
    def mse(x):
        return (x['stars']-x['prediction'])**2
    RMSE.loc[:,"mse"] = RMSE.apply(mse,axis=1)
    print('param:' ,param)
    print("RMSE : ", sqrt(RMSE['mse'].mean()))
'''



