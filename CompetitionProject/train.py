import pyspark
import os, sys, json, time, itertools, math

os.environ['PYSPARK_PYTHON'] = 'python3.6'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python3.6'

# hyper parameter
corated_threshold = 5 # both for item-based and user-based
pearson_correlation_threshold = 0.5
pearson_correlation_threshold_minus = -0.8


if __name__ == "__main__":

    start_time = time.time()

    train_file_path = "data/train_review.json" # input
    # TODO: don't forget to change the output file name each time change a method
    model_file_path = "out/model.out" # output

    conf = pyspark.SparkConf().setAppName("FinalProjectTrain").setMaster("local[*]")
    sc = pyspark.SparkContext(conf = conf)
    sc.setLogLevel("ERROR")

    rawRDD = sc.textFile(train_file_path).map(lambda x: json.loads(x)) \
        .map(lambda x: (x["business_id"], x["user_id"], x["stars"]))

    businessRDD = rawRDD.map(lambda x: x[0]).distinct().zipWithIndex()
    num_business = businessRDD.count()
    index_business_dict = businessRDD.map(lambda x: (x[1], x[0])).collectAsMap()

    userRDD = rawRDD.map(lambda x: x[1]).distinct().zipWithIndex()
    

    # cleanedRDD: ((business_idx, user_idx), stars)
    cleanedRDD = rawRDD.map(lambda x: (x[0], (x[1], x[2]))) \
        .join(businessRDD).map(lambda x: (x[1][0][0], (x[1][1], x[1][0][1]))) \
        .join(userRDD).map(lambda x: ((x[1][0][0], x[1][1]), x[1][0][1]))
    # deal with the fact that user review the same business multiple times
    # take the average rate
    cleanedRDD = cleanedRDD.groupByKey().mapValues(list).mapValues(lambda x: sum(x) / len(x))

    # for item-based: cleanedRDD: (business_idx, (user_idx, stars))
    cleanedRDD = cleanedRDD.map(lambda x: (x[0][0], (x[0][1], x[1])))
    
    business_user_score_dict_RDD = cleanedRDD.groupByKey().mapValues(dict)
    business_user_score_dict = business_user_score_dict_RDD.collectAsMap()
    business_user_score_dict_broadcast = sc.broadcast(business_user_score_dict)
    pairsRDD = sc.parallelize([i for i in range(num_business)]) \
            .flatMap(lambda x: [(x, i) for i in range(x + 1, num_business)])

    def filter_corated(idx1, idx2):
        user_set1 = set(business_user_score_dict_broadcast.value[idx1].keys())
        user_set2 = set(business_user_score_dict_broadcast.value[idx2].keys())
        return len(user_set1 & user_set2) >= corated_threshold

    def calculate_pearson(idx1, idx2):
        dict_1 = business_user_score_dict_broadcast.value[idx1]
        dict_2 = business_user_score_dict_broadcast.value[idx2]
        corated = set(dict_1.keys() & dict_2.keys())

        def calculate_avg_rating(dict):
            sum = 0
            for user in corated:
                sum += dict[user]
            return sum / len(corated)

        def update_dict(dict):
            res = {}
            avg = calculate_avg_rating(dict)
            for one in corated:
                res[one] = dict[one] - avg
            return res
        
        dict_1 = update_dict(dict_1)
        dict_2 = update_dict(dict_2)

        numerator, sum1, sum2 = 0, 0, 0
        for one in corated:
            numerator += dict_1[one] * dict_2[one]
            sum1 += pow(dict_1[one], 2)
            sum2 += pow(dict_2[one], 2)
        if numerator == 0:
            return 0
        return numerator / math.sqrt(sum1 * sum2)
    
    resultRDD = pairsRDD.filter(lambda x: filter_corated(x[0], x[1])) \
        .map(lambda x: (x[0], x[1], calculate_pearson(x[0], x[1]))) \
        .filter(lambda x: x[2] >= pearson_correlation_threshold or x[2] <= pearson_correlation_threshold_minus)
    
    result_pairs = resultRDD.collect()

    print("Total number of pairs in model: {}".format(len(result_pairs)))

    with open(model_file_path, 'w') as f:
        temp = {}
        for one in result_pairs:
            temp["b1"] = index_business_dict[one[0]]
            temp["b2"] = index_business_dict[one[1]]
            temp["sim"] = one[2]
            print(json.dumps(temp), file = f)

    print("Duration: {0:.2f}".format(time.time() - start_time))