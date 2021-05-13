import pyspark
import sys, json, operator

if __name__ == "__main__":

    # parse commandline argument
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    partition_type = sys.argv[3] # only "default" or "customized"
    partition_type = partition_type == "customized"
    n_partitions = int(sys.argv[4])
    n_threshold_reviews = int(sys.argv[5])

    conf = pyspark.SparkConf().setAppName("Task2").setMaster("local[*]")
    sc = pyspark.SparkContext(conf = conf)
    sc.setLogLevel("ERROR")

    result = {}

    reviewsRDD = sc.textFile(input_file).map(lambda review: json.loads(review)).map(lambda item: (item["business_id"], 1)).cache()

    if partition_type :
        def customized_hash(x):
            return hash(x[0])
        reviewsRDD = reviewsRDD.partitionBy(n_partitions, partitionFunc=customized_hash)

    result["n_partitions"] = reviewsRDD.getNumPartitions()
    print("number of partitions: ", result["n_partitions"])

    result["n_items"] = reviewsRDD.glom().map(lambda x: len(x)).collect()
    print(result["n_items"])

    result["result"] = reviewsRDD.reduceByKey(operator.add).filter(lambda x: x[1] > n_threshold_reviews).collect()
    # print(result["result"])

    with open(output_file, 'w') as f:
        json.dump(result, f, sort_keys=True)
