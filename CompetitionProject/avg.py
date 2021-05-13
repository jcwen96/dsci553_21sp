import pyspark
import os, sys, json, time
from operator import add

os.environ['PYSPARK_PYTHON'] = 'python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python3'

if __name__ == "__main__":

    start_time = time.time()

    train_file_path = "data/train_review.json"
    output_file_path = "out/business_avg.json"

    conf = pyspark.SparkConf().setAppName("FinalProjectPredict").setMaster("local[*]")
    sc = pyspark.SparkContext(conf = conf)
    sc.setLogLevel("ERROR")

    biz_star_RDD = sc.textFile(train_file_path).map(lambda x: json.loads(x)) \
        .map(lambda x: (x["business_id"], x["stars"]))
    
    all_avg = biz_star_RDD.map(lambda x: x[1]).reduce(add) / biz_star_RDD.count()

    biz_avg_RDD = biz_star_RDD.groupByKey().mapValues(lambda x: sum(x) / len(x))
    avg_dict = biz_avg_RDD.collectAsMap()

    avg_dict["UNK"] = all_avg

    with open(output_file_path, 'w') as f:
        json.dump(avg_dict, f)
    
    
    print("Duration: {0:.2f}".format(time.time() - start_time))