# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:37:11 2021

@author: 13451
"""


from pyspark import SparkContext
import time
import sys
import copy

args = sys.argv[:]

threshold = int(args[1])
input_file = args[2]
output_file_between = args[3]
output_file_community = args[4]
sc = SparkContext(master='local[*]',appName='AS4T2')
sc.setLogLevel("ERROR")

start = time.time()
raw = sc.textFile(input_file)
header = raw.first()
data = raw.filter(lambda x: x != header).map(lambda x: x.split(','))

def into_set(x):
    return set(x)
users = data.groupByKey().mapValues(into_set)
user_dict = users.collectAsMap()

def create_edge(row):
    nodes = set() 
    edge_lst = list() 
    user = row[0]
    biz = user_dict[row[0]]
    for other in user_dict.keys():
        if other != user:
            corated = user_dict[other].intersection(biz) 
            if len(corated) >= threshold:
                nodes.add(other)
                edge_lst.append((user,other))
    if len(nodes)==0 and len(edge_lst)==0:
        return (0,0)
    else:
        return (edge_lst,nodes)

edgelist = users.map(create_edge).filter(lambda x:x[0]!=0)

vertices = edgelist.flatMap(lambda x: x[1]).distinct()
edges = edgelist.flatMap(lambda x: x[0])

edge_map = edges.groupByKey().mapValues(into_set).collectAsMap()

######### betweeness ##########
def bfs(row):
    begin = row
    visited = set()
    visited.add(begin)
    curlevel_1 =set()
    curlevel_1.add(begin)
    nextlevel_1 =set()  #save point 
    structure = [{begin}]
    while len(curlevel_1)!=0:
        level_map = dict()
        cur_visited = set()
        for point in curlevel_1:
            for next_point in edge_map[point]:
                if next_point not in visited:
                    nextlevel_1.add(next_point)
                    cur_visited.add(next_point)
                    level_map.setdefault(next_point,set())
                    level_map[next_point].add(point)
        if len(level_map)!=0:
            structure.append(level_map)
        visited = visited.union(cur_visited)
        curlevel_1 = nextlevel_1
        nextlevel_1 = set()

    ans = []
    i = len(structure)-1
    value_map = {}
    while i>=1:
        #reversed tranverse
        prelevel_2 = structure[i-1]
        curlevel_2 = structure[i]
        if i>=2:
            for point in curlevel_2.keys():
                value_map.setdefault(point,1)
                sum_short_path = 0
                for prepoint in curlevel_2[point]:
                    sum_short_path+= len(prelevel_2[prepoint]) 
                for prepoint in curlevel_2[point]:
                    edge_val = value_map[point]*len(prelevel_2[prepoint])/sum_short_path
                    ans.append((tuple(sorted([point,prepoint])),edge_val))
                    value_map.setdefault(prepoint,1)
                    value_map[prepoint]+= edge_val
        else:
            for point in curlevel_2.keys():
                for prepoint in curlevel_2[point]:
                    value_map.setdefault(point,1)
                    edge_val = value_map[point]
                    ans.append((tuple(sorted([point,prepoint])),edge_val))
                    value_map.setdefault(prepoint,1)
                    value_map[prepoint]+= edge_val
        i-=1
    
    return ans

betweeness = vertices.flatMap(bfs).reduceByKey(lambda x, y: x+y)\
    .map(lambda x: (x[0],x[1]/2)).sortBy(lambda x: (-x[1], x[0][0], x[0][1])).collect()
        
f = open(output_file_between, 'w')
for i in betweeness:
    f.write("(\'" + str(i[0][0] + "\', \'" + str(i[0][1]) + "\'), " + str(round(i[1],5)) + "\n"))
f.close()

######## cut to find communities ###########
m = edges.count()/2
remain_edges = m

def simple_bfs_community(vertex): #for single point
    begin = vertex
    visited = set()
    visited.add(begin)
    curlevel = set()
    curlevel.add(begin)
    nextlevel= set() 
    community = set()
    community.add(begin)
    while len(curlevel)!=0:
        for point in curlevel:
            for next_point in edge_map[point]:
                if next_point not in visited:
                    community.add(next_point)
                    nextlevel.add(next_point)
                    visited.add(next_point)
        curlevel = nextlevel
        nextlevel = set()
    return community
    
def get_global_communities(vertices): # for global
    communities = []
    visited = set()
    for vertex in vertices:
        if vertex not in visited:
            community = simple_bfs_community(vertex)
            visited = visited.union(community)
            communities.append(sorted(list(community)))      
    return communities

def compute_modularity(communities):
    modularity = 0.0
    denominator = 2 * m
    for community in communities:
        for i in community:
            for j in community:
                if j!=i:
                    actual = 1 if j in edge_map[i] else 0
                    expected = (len(edge_map[i]) * len(edge_map[j])) / denominator
                    modularity += (actual - expected)
    return modularity / denominator

def cut_graph(edges):
    for edge in edges:
        edge_map[edge[0]].remove(edge[1])
        edge_map[edge[1]].remove(edge[0])
        
vertices_list = vertices.collect()
best_modularity =-1
increasing_check = 0
last_modularity = -1
while remain_edges > 0:
    communities = get_global_communities(vertices_list)
    modularity = compute_modularity(communities)
    if modularity <= last_modularity:
        increasing_check +=1
    else:
        increasing_check = 0
    if modularity > best_modularity:
        best_modularity = modularity
        modular_community = copy.deepcopy(communities)
    min_cut = vertices.flatMap(bfs).reduceByKey(lambda x, y: x+y)\
        .map(lambda x: (x[1]/2,x[0])).groupByKey().mapValues(list).sortBy(lambda x: (-x[0])).first()[1]
    cut_graph(min_cut)
    remain_edges -= len(min_cut)
    last_modularity = modularity
    ### break situation
    if increasing_check>=10:
        break

best_communities = sc.parallelize(modular_community).sortBy(lambda x: (len(x), x)).collect()

f = open(output_file_community, 'w')
for row in best_communities:
    line = ""
    for node in row:
        line = line + "'" + str(node) + "', "
    line = line[:-2]
    f.write(line)
    f.write('\n')
f.close()

end = time.time()
print("Duration:"+ str(end-start))

