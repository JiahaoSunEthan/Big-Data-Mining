# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 13:40:09 2021

@author: 13451
"""


from blackbox import BlackBox
import sys
import time
import random

args = sys.argv[:]
input_file = args[1]
stream_size = int(args[2])
num_of_asks = int(args[3])
output_file = args[4]

start = time.time()
reservoir = [0] * 100
random.seed(553)
seqnum = 0

def reservoir_sampling(stream_users,ask):
    global reservoir
    global seqnum
    if ask == 0:
        for index in range(stream_size):
            reservoir[index] = stream_users[index]
        seqnum += 100
    else:
        for user in stream_users:
            seqnum += 1
            if random.random() <= stream_size/seqnum:
                x = random.randint(0,99)  # chossing the location in list
                reservoir[x] = user
    f.write(str(seqnum) + ',' + str(reservoir[0]) + ',' + str(reservoir[20]) + ',' + str(reservoir[40]) + ',' + str(reservoir[60]) + ',' + str(reservoir[80]) + '\n')
    

f = open(output_file, "w")
f.write("seqnum,0_id,20_id,40_id,60_id,80_id\n")

bx = BlackBox()
for ask in range(num_of_asks):
    stream_users = bx.ask(input_file, stream_size)
    reservoir_sampling(stream_users,ask)
f.close()
print("Duration : ", time.time() - start)