# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 16:07:20 2021

@author: 13451
"""


from graphframes import GraphFrame
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
import time
import os
import sys

#os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell")
#os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 pyspark-shell"
args = sys.argv[:]

threshold = int(args[1])
input_file = args[2]
output_file = args[3]
sc = SparkContext(master='local[*]',appName='AS4T1')
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)

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

vertices = edgelist.flatMap(lambda x: x[1]).distinct().map(lambda x: Row(x))
edges = edgelist.flatMap(lambda x: x[0])

v = sqlContext.createDataFrame(vertices,["id"])
e = sqlContext.createDataFrame(edges, ["src", "dst"])

gf = GraphFrame(v, e)
lpa_df = gf.labelPropagation(maxIter=5)
output = lpa_df.rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(lambda x: sorted(list(x))) \
    .sortBy(lambda x: (len(x[1]), x[1])).map(lambda x: tuple(x[1])).collect()

f = open(output_file, 'w')
for row in output:
    line = ""
    for node in row:
        line = line + "'" + str(node) + "', "
    line = line[:-2]
    f.write(line)
    f.write('\n')
f.close()

end = time.time()
print("Duration:"+ str(end-start))
