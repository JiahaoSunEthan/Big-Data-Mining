from surprise import SVD
from surprise import Dataset
from surprise import Reader
import surprise
import time
import pandas as pd
import numpy
from pyspark.sql import SQLContext

from pyspark import SparkContext, SparkConf
import json
from itertools import combinations
import math

#os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
#os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

if __name__ == '__main__':
    start = time.time()
    train_file = "train_review.json"

    conf = SparkConf().setMaster("local[*]")
    conf.set("spark.executor.memory","4g")
    conf.set("spark.driver.memory","4g")
    sc = SparkContext.getOrCreate(conf)
    cnx = SQLContext(sc)
    sc.setLogLevel("ERROR")

    TRAIN = sc.textFile(train_file) \
    .map(lambda line: json.loads(line)) \
    .map(lambda x: (x["user_id"], x["business_id"], x["stars"]))

    sparkDF = cnx.createDataFrame(TRAIN, schema = ["uid","bid","stars"])

    df = sparkDF.toPandas()

    reader = Reader(rating_scale = (0,5))

    trainset = Dataset.load_from_df(df, reader).build_full_trainset()

    algo = SVD()
    MODEL = algo.fit(trainset)
    surprise.dump.dump("svd.model", algo = MODEL)

    print("Duration: %d" %(time.time() - start))