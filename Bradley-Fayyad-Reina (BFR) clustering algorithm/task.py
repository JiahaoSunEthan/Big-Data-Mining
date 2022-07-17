# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 20:53:31 2021

@author: 13451
"""

import math
import sys
from sklearn.cluster import KMeans
import numpy as np
import time
import random
import itertools
import os
os.environ["OMP_NUM_THREADS"] = "1"

input_file = sys.argv[1]
num_clusters = int(sys.argv[2])
output_file = sys.argv[3]
start = time.time()

f1 = open(input_file, "r")
data = []
for line in f1.readlines():
    data.append(list(map(lambda s: float(s),line[:-1].split(','))))
f1.close()

#Step 1. Load 20% of the data randomly
rand_size = round(len(data) * 0.2)
slice_start = 0
data_np = np.array(data)
np.random.shuffle(data_np)
data_s1 = data_np[slice_start:slice_start+rand_size]
dim = len(data_np[0])-2

#Step 2. Run K-Means (e.g., from sklearn) with a large K
# on the data in memory using the Euclidean distance as the similarity measurement
first_load = data_s1[:,2:]
kmeans = KMeans(n_clusters=5 * num_clusters, random_state=100)
clusters = kmeans.fit_predict(first_load) 

def generate_cluster_dict(clusters,data): 
    clusters_dict = dict() 
    for i in range(len(clusters)):
        point = data[i]
        clusterid = clusters[i]
        if clusterid in clusters_dict:
            clusters_dict[clusterid].append(point)
        else:
            clusters_dict[clusterid] = [point]
    return clusters_dict

clusters_dict = generate_cluster_dict(clusters,data_s1)

#Step 3. In the K-Means result from Step 2, move all the clusters that 
# contain only one point to RS (outliers)
retained_set = dict() ######################## wait to optimize
for key in clusters_dict.keys():
    if len(clusters_dict[key]) == 1:
        point = clusters_dict[key][0]
        index = np.where(data_s1[:,0]== point[0]) #locate the data point index
        retained_set[point[0]] = point
        data_s1 = np.delete(data_s1,index,axis=0) #updated data_s1

#Step 4. Run K-Means again to cluster the rest of the data points 
# with K = the number of input clusters.
second_load = data_s1[:,2:]
kmeans = KMeans(n_clusters = num_clusters, random_state=100)
clusters = kmeans.fit_predict(second_load)
clusters_dict = generate_cluster_dict(clusters,data_s1)

#Step 5. Use the K-Means result from Step 4 to generate 
#the DS clusters (i.e., discard their points and generate statistics).
discard_set = dict()
for key in clusters_dict.keys(): 
#structure of discard_set dict cluster id : [member_list of index, n, sum, sqaure_sum,centroid]
    discard_set[key] = []
    member_list =[]
    for point in clusters_dict[key]:
        member_list.append(point[0])
    discard_set[key].append(member_list)
    discard_set[key].append(len(member_list))
    discard_set[key].append(np.sum(np.array(clusters_dict[key])[:,2:], axis=0))
    discard_set[key].append(np.sum(np.square(np.array(clusters_dict[key])[:,2:]), axis=0))
    discard_set[key].append(np.sum(np.array(clusters_dict[key])[:,2:], axis=0)/len(member_list))

#The initialization of DS has finished, so far, you have K numbers of DS clusters (from Step 5)
# and some numbers of RS (from Step 3).

#Step 6. Run K-Means on the points in the RS with a large K (e.g., 5 times of the number of the input clusters) 
#to generate CS (clusters with more than one points) and RS (clusters with only one point).
retained_set_points = [] ######################## wait to optimize
for key in retained_set.keys():
    retained_set_points.append(retained_set[key])

rs_points_array = np.array(retained_set_points)[:,2:]
kmeans = KMeans(n_clusters=int(len(retained_set_points) / 2 + 1))
cs_clusters = generate_cluster_dict(kmeans.fit_predict(rs_points_array),np.array(retained_set_points))

compression_set = dict()
for key in cs_clusters.keys():
    if len(cs_clusters[key]) > 1:
        compression_set[key] = []
        member_list =[]
        for point in cs_clusters[key]:
            member_list.append(point[0])
            del retained_set[point[0]] #delete compressed data points in RS
        compression_set[key].append(member_list)
        compression_set[key].append(len(member_list))
        compression_set[key].append(np.sum(np.array(cs_clusters[key])[:,2:], axis=0))
        compression_set[key].append(np.sum(np.square(np.array(cs_clusters[key])[:,2:]), axis=0))
        compression_set[key].append(np.sum(np.array(cs_clusters[key])[:,2:], axis=0)/len(member_list))

def output_intermediate(f, round_num):
    ds_points_count = 0
    cs_clusters_count = 0
    cs_points_count = 0
    for key in discard_set.keys():
        ds_points_count += discard_set[key][1]
    for key in compression_set.keys():
        cs_clusters_count += 1
        cs_points_count += compression_set[key][1]
    rs_points_count = len(retained_set)
    f.write(
        "Round " + str(round_num) + ": " + str(ds_points_count) + "," + str(cs_clusters_count) + "," + str(
            cs_points_count) + "," + str(rs_points_count) + "\n")

f2 = open(output_file, "w")
f2.write("The intermediate results:\n")
round_num =1
output_intermediate(f2, round_num)


def compute_Mahalanobis_Distance(array1,array2,std_dev):
    mahalanobis_distance = 0
    for d in range(0, dim):
        mahalanobis_distance += ((array1[d] - array2[d]) / std_dev[d]) ** 2
    mahalanobis_distance = math.sqrt(mahalanobis_distance)
    return mahalanobis_distance

def merge_cs_clusters(cs1_key, cs2_key):
    global compression_set
    compression_set[cs1_key][0].extend(compression_set[cs2_key][0])
    compression_set[cs1_key][1] = compression_set[cs1_key][1] + compression_set[cs2_key][1]
    compression_set[cs1_key][2] = compression_set[cs1_key][2] + compression_set[cs2_key][2]
    compression_set[cs1_key][3] = compression_set[cs1_key][3] + compression_set[cs2_key][3]
    compression_set[cs1_key][4] = compression_set[cs1_key][2]/compression_set[cs1_key][1]
    del compression_set[cs2_key]

def merge_cs_with_ds(cs_key, ds_key):
    global compression_set
    global discard_set
    discard_set[ds_key][0].extend(compression_set[cs_key][0])
    discard_set[ds_key][1] = discard_set[ds_key][1] + compression_set[cs_key][1]
    discard_set[ds_key][2] = discard_set[ds_key][2] + compression_set[cs_key][2]
    discard_set[ds_key][3] = discard_set[ds_key][3] + compression_set[cs_key][3]
    discard_set[ds_key][4] = discard_set[ds_key][2]/discard_set[ds_key][1]
    del compression_set[cs_key]

def RankFirst(elem):
    return elem[0]

def RankSecond(elem):
    return elem[1]

def check_cluster_centroid_distance(cluster_set): # for compression set, don't need to recursively merge
    merge_waitlist=[]
    for combination in itertools.combinations(cluster_set.keys(), 2):
        N_1 = cluster_set[combination[0]][1]
        SUM_d_1 = cluster_set[combination[0]][2]
        SUMSQ_d_1 = cluster_set[combination[0]][3]
        centroid_1 = cluster_set[combination[0]][4]
        std_1 = np.sqrt(SUMSQ_d_1/N_1 - np.square(SUM_d_1/N_1))
        
        N_2 = cluster_set[combination[1]][1]
        SUM_d_2 = cluster_set[combination[1]][2]
        SUMSQ_d_2 = cluster_set[combination[1]][3]
        centroid_2 = cluster_set[combination[1]][4]
        std_2 = np.sqrt(SUMSQ_d_2/N_2 - np.square(SUM_d_2/N_2))
        
        distance = min(compute_Mahalanobis_Distance(list(centroid_1),list(centroid_2),list(std_1)),compute_Mahalanobis_Distance(list(centroid_1),list(centroid_2),list(std_2)))
        
        if distance < 2*math.sqrt(dim):
            merge_waitlist.append((combination,distance))
    merge_waitlist.sort(key = RankSecond,reverse = False)
    return merge_waitlist

def find_closest_DS_cluster(cs_key):
    N_1 = compression_set[cs_key][1]
    SUM_d_1 = compression_set[cs_key][2]
    SUMSQ_d_1 = compression_set[cs_key][3]
    centroid_1 = compression_set[cs_key][4]
    std_1 = np.sqrt(SUMSQ_d_1/N_1 - np.square(SUM_d_1/N_1))
    
    closest_distance = float('inf')
    closest_cluster = -1
    for key in discard_set.keys():
        N_2 = discard_set[key][1]
        SUM_d_2 = discard_set[key][2]
        SUMSQ_d_2 = discard_set[key][3]
        centroid_2 = discard_set[key][4]
        std_2 = np.sqrt(SUMSQ_d_2/N_2 - np.square(SUM_d_2/N_2))
        
        distance = min(compute_Mahalanobis_Distance(list(centroid_1),list(centroid_2),list(std_1)),compute_Mahalanobis_Distance(list(centroid_1),list(centroid_2),list(std_2)))
        if distance <= closest_distance:
            closest_distance = distance
            closest_cluster = key
    
    return closest_distance,closest_cluster

#Step 7. Load another 20% of the data randomly
while round_num<=4:
    slice_start += rand_size
    round_num+=1
    if round_num ==5:
        slice_end = len(data)
    else:
        slice_end = slice_start+ rand_size
    data_s2 = data_np[slice_start:slice_end]

#Step 8. For the new points, compare them to each of the DS using the Mahalanobis Distance and assign
#them to the nearest DS clusters if the distance is < 2√𝑑.
#Step 9. For the new points that are not assigned to DS clusters, using the Mahalanobis Distance and assign
#the points to the nearest CS clusters if the distance is < 2√𝑑
#Step 10. For the new points that are not assigned to a DS cluster or a CS cluster, assign them to RS.
    for i in range(len(data_s2)):
        newpoint = data_s2[i]
        newpoint_data = newpoint[2:]
        closest_distance = float('inf')
        closest_cluster = -1
        for key in discard_set.keys(): #compute cloesest distance
            N = discard_set[key][1]
            SUM_d = discard_set[key][2]
            SUMSQ_d = discard_set[key][3]
            centroid = discard_set[key][4]
            std = np.sqrt(SUMSQ_d/N - np.square(SUM_d/N))
            distance = compute_Mahalanobis_Distance(list(newpoint_data),list(centroid),list(std))
            if distance <= closest_distance:
                closest_distance = distance
                closest_cluster = key
        
        if closest_distance < 2*math.sqrt(dim):
            # Step 8
            discard_set[closest_cluster][0].append(newpoint[0])
            discard_set[closest_cluster][1] +=1
            discard_set[closest_cluster][2] = discard_set[closest_cluster][2] + newpoint_data
            discard_set[closest_cluster][3] = discard_set[closest_cluster][3] + np.square(newpoint_data)
            discard_set[closest_cluster][4] = discard_set[closest_cluster][2]/discard_set[closest_cluster][1]
        else:
            closest_distance = float('inf')
            closest_cluster = -1
            for key in compression_set.keys(): #compute closest distance
                N = compression_set[key][1]
                SUM_d = compression_set[key][2]
                SUMSQ_d = compression_set[key][3]
                centroid = compression_set[key][4]
                std = np.sqrt(SUMSQ_d/N - np.square(SUM_d/N))
                distance = compute_Mahalanobis_Distance(list(newpoint_data),list(centroid),list(std))
                if distance <= closest_distance:
                    closest_distance = distance
                    closest_cluster = key
            
            if closest_distance < 2*math.sqrt(dim):
                 # Step 9
                compression_set[closest_cluster][0].append(newpoint[0])
                compression_set[closest_cluster][1] +=1
                compression_set[closest_cluster][2] = compression_set[closest_cluster][2] + newpoint_data
                compression_set[closest_cluster][3] = compression_set[closest_cluster][3] + np.square(newpoint_data)
                compression_set[closest_cluster][4] = compression_set[closest_cluster][2]/compression_set[closest_cluster][1]
            else:
                # Step 10
                retained_set[newpoint[0]] = newpoint

#Step 11. Run K-Means on the RS with a large K (e.g., 5 times of the number of the input clusters) to
#generate CS (clusters with more than one points) and RS (clusters with only one point)
    retained_set_points = [] ######################## wait to optimize
    for key in retained_set.keys():
        retained_set_points.append(retained_set[key])
    new_points_array = np.array(retained_set_points)[:,2:]
    kmeans = KMeans(n_clusters=int(len(retained_set_points) / 2 + 1), random_state=0)
    cs_clusters = generate_cluster_dict(kmeans.fit_predict(new_points_array),np.array(retained_set_points))

    for key in cs_clusters.keys():
        if len(cs_clusters[key]) > 1:
            new_key = random.random()
            while new_key in compression_set.keys():
                new_key = random.random()
            compression_set[new_key] = []
            member_list =[]
            for point in cs_clusters[key]:
                member_list.append(point[0])
                del retained_set[point[0]] #delete compressed data points in RS
            compression_set[new_key].append(member_list)
            compression_set[new_key].append(len(member_list))
            compression_set[new_key].append(np.sum(np.array(cs_clusters[key])[:,2:], axis=0))
            compression_set[new_key].append(np.sum(np.square(np.array(cs_clusters[key])[:,2:]), axis=0))
            compression_set[new_key].append(np.sum(np.array(cs_clusters[key])[:,2:], axis=0)/len(member_list))

# Step 12. Merge CS clusters that have a Mahalanobis Distance < 2√𝑑
    merge_waitlist = check_cluster_centroid_distance(compression_set) 
    merged = set()
    for combination in merge_waitlist:
        if combination[0][0] not in merged and combination[0][1] not in merged:
            merge_cs_clusters(combination[0][0], combination[0][1])
            merged.add(combination[0][0])
            merged.add(combination[0][1])
    
    if round_num ==5:
        compression_set_keys = list(compression_set.keys()) #can't do the iteration and modification at the same time
        for cs_key in compression_set_keys:
            distance,ds_key = find_closest_DS_cluster(cs_key)
            if distance < 2*math.sqrt(dim):
                merge_cs_with_ds(cs_key, ds_key)
    
    output_intermediate(f2, round_num)
    print('round'+str(round_num)+'finished')


f2.write("\nThe clustering results: ")
cluster_result = []
for key in discard_set.keys():
    for point in discard_set[key][0]:
        cluster_result.append((point,key))
for key in compression_set.keys():
    for point in compression_set[key][0]:
        cluster_result.append((point,-1))
for key in retained_set.keys():
    cluster_result.append((key,-1))
cluster_result.sort(key = RankFirst, reverse = False)
for point in cluster_result:
    f2.write("\n" + str(int(point[0])) + "," + str(point[1]))

f2.close()
print("Duration : ", time.time() - start)
    
    
    
    


