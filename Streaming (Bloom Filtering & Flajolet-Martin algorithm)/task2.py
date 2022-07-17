# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 14:14:29 2021

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
actual_total = 0
estimated_total = 0

def myhashs(s):
    hash_a = [332, 993, 568, 476, 380, 10, 991, 883, 517, 430, 552, 830, 805, 775, 726, 527]
    hash_b = [572, 403, 428, 621, 786, 451, 790, 335, 970, 97, 88, 811, 71, 991, 601, 842]
    hash_p = [6649385, 6475799, 4416863, 8564383, 5955983, 4433527, 380121, 1127229, 738500, 2007533, 6623519, 9440624,668655, 2632966, 1674740, 9491576]
    result= [(hash_a[i]*int(binascii.hexlify(s.encode('utf8')),16)+ hash_b[i])%hash_p[i]%(2**10) for i in range(16)]
    return result

def combine_est(lst):
    average_lst=[]
    i,j = 0,3
    while j<len(lst)-1:
        sum_val=0
        num=0
        for index in range(i,j+1):
            num+=1
            sum_val+=lst[index]
        average_lst.append(sum_val/num)
        i+=4
        j+=4
    average_lst = sorted(average_lst)
    return round((average_lst[1]+average_lst[2])/2)
        
def flajolet_martin(stream_users, ask):
    global actual_total
    global estimated_total
    estimations = []
    all_hash_values = []
    for user_id in stream_users:
        hash_values = myhashs(user_id)
        all_hash_values.append(hash_values)
    for i in range(len(all_hash_values[0])):
        max_traling_zeroes = -1
        for hash_value in all_hash_values:
            trailing_zeros = len(bin(hash_value[i])[2:]) - len(bin(hash_value[i])[2:].rstrip("0"))
            if trailing_zeros > max_traling_zeroes:
                max_traling_zeroes = trailing_zeros
        estimations.append(2 ** max_traling_zeroes)
    estimated_count = combine_est(estimations)
    actual_count = len(set(stream_users))
    estimated_total += estimated_count
    actual_total += actual_count
    f.write(str(ask) + "," + str(actual_count) + "," + str(estimated_count) + "\n")


f = open(output_file, "w")
f.write("Time,Ground Truth,Estimation\n")

bx = BlackBox()
for ask in range(num_of_asks):
    stream_users = bx.ask(input_file, stream_size)
    flajolet_martin(stream_users, ask)
f.close()
print(float(estimated_total / actual_total))
print("Duration : ", time.time() - start)