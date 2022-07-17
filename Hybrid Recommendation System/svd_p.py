from surprise import SVD
from surprise import Dataset
from surprise import Reader
import surprise
import time
import pandas as pd
import numpy
#from pyspark import SparkContext, SparkConf
import json
import sys
import os
#from pyspark.sql import SQLContext

if __name__ == "__main__":
    start = time.time()

    test_file = sys.argv[1]
    output_file = sys.argv[2]
    input_model_path = "svd.model"

    svd_MODEL = surprise.dump.load(input_model_path)[1]
    print("Duration: %d" %(time.time() - start))

    res = list()
    with open(test_file, "r") as f_in:
        for line in f_in.readlines():
            line = json.loads(line)
            line = (line['user_id'],line['business_id'])
            est = svd_MODEL.predict(uid=line[0],iid=line[1])
            est = [i for i in est]
            est = str(est[3])
            res.append((line[0],line[1],est))

    with open(output_file, "w") as f:
        for line in res:
            f.write(
                json.dumps({"user_id":line[0],"business_id":line[1],"stars":line[2]}) + "\n"
            )
    
    print("Duration: %d" %(time.time() - start))
    





