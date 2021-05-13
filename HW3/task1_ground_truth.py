import pyspark
import sys, json, time


# hyper parameter
similarity_threshold = 0.05


if __name__ == "__main__":

    start_time = time.time()

    # parse commandline argument
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    conf = pyspark.SparkConf().setAppName("Task1_Ground_Truth").setMaster("local[*]")
    sc = pyspark.SparkContext(conf = conf)
    sc.setLogLevel("ERROR")

    rawRDD = sc.textFile(input_file_path) \
        .map(lambda x: json.loads(x)) \
        .map(lambda x: [x["business_id"], x["user_id"]])

    business_user_RDD = rawRDD.groupByKey().mapValues(set)

    business_pairs_RDD = business_user_RDD.cartesian(business_user_RDD) \
        .filter(lambda x: x[0][0] < x[1][0])

    def compute_jaccard(user_set1, user_set2):
        common = len(user_set1.intersection(user_set2))
        total = len(user_set1) + len(user_set2) - common
        return common / total

    resultRDD =  business_pairs_RDD \
        .map(lambda x: (x[0][0], x[1][0], compute_jaccard(x[0][1], x[1][1]))) \
        .filter(lambda x: x[2] >= similarity_threshold)

    results = resultRDD.collect()
    print("total similar pairs:", len(results))

    with open(output_file_path, 'w') as f:
        temp = {}
        for one in results:
            temp["b1"] = one[0]
            temp["b2"] = one[1]
            temp["sim"] = one[2]
            print(json.dumps(temp), file = f)
            

    end_time = time.time()
    print("Duration: {0:.2f}".format(end_time - start_time))