# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 12:46:28 2021

@author: 13451
"""


from blackbox import BlackBox
import binascii
import sys
import time

args = sys.argv[:]
input_file = args[1]
stream_size = int(args[2])
num_of_asks = int(args[3])
output_file = args[4]

start = time.time()
filter_bitarray = [0] * 69997
user_set = set()

def myhashs(s):
    hash_func_index = [(1,111),(5,222),(9,333),(13,444),(17,555),(21,666)]
    result= [(a*int(binascii.hexlify(s.encode('utf8')),16)+b)%69997 for (a,b) in hash_func_index]
    return result

def bloom_filter(stream_users,ask_index):
    global filter_bitarray
    global user_set
    tn,fp = 0,0
    for user_id in stream_users:
        hash_values = myhashs(user_id)
        num_hit = 0
        for value in hash_values:
            if filter_bitarray[value] ==1:
                num_hit+=1
            else:
                filter_bitarray[value] =1
            if user_id not in user_set:
                if num_hit == len(hash_values):
                    fp += 1  # False positive: x not in S, but identified as in S
                else:
                    tn += 1
        user_set.add(user_id)
    
    if fp == 0 and tn == 0:
        fpr = 0.0
    else:
        fpr = float(fp / float(fp+ tn))
    f.write(str(ask_index) + "," + str(fpr) + "\n")
        
f = open(output_file, "w")
f.write("Time,FPR\n")

bx = BlackBox()
for ask in range(num_of_asks):
    stream_users = bx.ask(input_file, stream_size)
    bloom_filter(stream_users,ask)
f.close()
print("Duration : ", time.time() - start)