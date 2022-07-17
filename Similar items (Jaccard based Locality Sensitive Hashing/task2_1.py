# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 22:16:37 2021

@author: 13451
"""


from pyspark import SparkContext
from math import sqrt
import time
import sys

args = sys.argv[:]

filepath_train = args[1]
filepath_test = args[2]
output_file = args[3]
sc = SparkContext(master='local[*]',appName='AS3T21')

start = time.time()
train_rdd = sc.textFile(filepath_train)
train_header = train_rdd.first()
train = train_rdd.filter(lambda x: x != train_header).map(lambda x: x.split(','))

test_rdd = sc.textFile(filepath_test)
test_header = test_rdd.first()
test = test_rdd.filter(lambda x: x != test_header).map(lambda x: x.split(','))

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
'''
def compute_rating(row,n):
    user = row[0]
    business = row[1]
    real_rating = row[2]
    if business not in business_map:
        if user not in user_map: #### new business & new user
            return [user,business,real_rating,'2.5']
        else: ### new business & old user
            return [user,business,real_rating,str(user_avg_map[user])]
    else:
        if user not in user_map: ### old business & new user
            return [user,business,real_rating,str(business_avg_map[business])]
        else:  ### old business & old user
            neighbour = []
            for bus_neigh in user_map[user].keys():
                if compute_pearson_correlation(business,bus_neigh)> 0: #only consider pearson coef >0
                    neighbour.append((bus_neigh,compute_pearson_correlation(business,bus_neigh)))
            if len(neighbour) ==0:
                default_rating = (user_avg_map[user]+ business_avg_map[business])/2
                return [user,business,real_rating,str(default_rating)]
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
            return [user,business,real_rating,str(rating_numerator/rating_denominator) ]
        

def cutscore(row):
       if float(row[3]) > 5:
           row[3] = '5.0'
       elif float(row[3]) < 1:
           row[3] = '1.0'
       else:
           row[3] = row[3]
       return (row[0],row[1],row[2],row[3])  
'''

def compute_rating(row,n):
    user = row[0]
    business = row[1]
    if business not in business_map:
        if user not in user_map: #### new business & new user
            return [user,business,'2.5']
        else: ### new business & old user
            return [user,business,str(user_avg_map[user])]
    else:
        if user not in user_map: ### old business & new user
            return [user,business,str(business_avg_map[business])]
        else:  ### old business & old user
            neighbour = []
            for bus_neigh in user_map[user].keys():
                if compute_pearson_correlation(business,bus_neigh)> 0: #only consider pearson coef >0
                    neighbour.append((bus_neigh,compute_pearson_correlation(business,bus_neigh)))
            if len(neighbour) ==0:
                default_rating = (user_avg_map[user]+ business_avg_map[business])/2
                return [user,business,str(default_rating)]
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
            return [user,business,str(rating_numerator/rating_denominator)]
        

def cutscore(row):
       if float(row[2]) > 5:
           row[2] = '5.0'
       elif float(row[2]) < 1:
           row[2] = '1.0'
       else:
           row[2] = row[2]
       return (row[0],row[1],row[2])
'''
def cutscore(row):
    if float(row[3]) <-3:
        row[3] = '1.0'
    elif float(row[3]) < -1:
        row[3] = '2.0'
    elif float(row[3])< 1:
        row[3] = '3.0'
    elif float(row[3])< 3:
        row[3] = '4.0'
    else:
        row[3] = '5.0'
    return (row[0],row[1],row[2],row[3])
'''

test_predict = test.map(lambda row:compute_rating(row,8)).map(cutscore)

'''
diff = test_predict.map(lambda r: abs(float(r[2]) - float(r[3])))
diff01 = diff.filter(lambda x: 0 <= x < 1)
diff12 = diff.filter(lambda x: 1 <= x < 2)
diff23 = diff.filter(lambda x: 2 <= x < 3)
diff34 = diff.filter(lambda x: 3 <= x < 4)
diff4 = diff.filter(lambda x: 4 <= x)
MSE = diff.map(lambda x: x**2).mean()
RMSE = sqrt(MSE)
print(RMSE)
'''

with open(output_file, 'w') as f:
    f.write("user_id,business_id,prediction\n")
    for line in test_predict.collect():
        f.write(str(line[0]) + "," + str(line[1]) + "," + str(line[2]) + "\n")
m = time.time()-start
print("Duration:"+ str(m))






                
            
        

    

