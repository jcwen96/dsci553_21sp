import pyspark
import sys, json, time, itertools, math

# hyper parameter
corated_threshold = 3 # both for item-based and user-based
jaccard_similarity_threshold = 0.01 # for user-based


if __name__ == "__main__":

    start_time = time.time()

    # parse commandline argument
    train_file_path = sys.argv[1] # input
    model_file_path = sys.argv[2] # output
    cf_type = sys.argv[3]
    cf_type = cf_type == "item_based"

    conf = pyspark.SparkConf().setAppName("Task3Train").setMaster("local[*]")
    sc = pyspark.SparkContext(conf = conf)
    sc.setLogLevel("ERROR")

    rawRDD = sc.textFile(train_file_path).map(lambda x: json.loads(x)) \
        .map(lambda x: (x["business_id"], x["user_id"], x["stars"]))

    businessRDD = rawRDD.map(lambda x: x[0]).distinct().zipWithIndex()
    if cf_type:
        num_business = businessRDD.count()
        index_business_dict = businessRDD.map(lambda x: (x[1], x[0])).collectAsMap()

    userRDD = rawRDD.map(lambda x: x[1]).distinct().zipWithIndex()
    if not cf_type:
        num_user = userRDD.count()
        index_user_dict = userRDD.map(lambda x: (x[1], x[0])).collectAsMap()
    

    # cleanedRDD: ((business_idx, user_idx), stars)
    cleanedRDD = rawRDD.map(lambda x: (x[0], (x[1], x[2]))) \
        .join(businessRDD).map(lambda x: (x[1][0][0], (x[1][1], x[1][0][1]))) \
        .join(userRDD).map(lambda x: ((x[1][0][0], x[1][1]), x[1][0][1]))
    # deal with the fact that user review the same business multiple times
    # take the average rate
    cleanedRDD = cleanedRDD.groupByKey().mapValues(list).mapValues(lambda x: sum(x) / len(x))

    if cf_type:
        # for item-based: cleanedRDD: (business_idx, (user_idx, stars))
        cleanedRDD = cleanedRDD.map(lambda x: (x[0][0], (x[0][1], x[1])))
    else :
        # for user-based: cleanedRDD: (user_idx, (business_idx, stars))
        cleanedRDD = cleanedRDD.map(lambda x: (x[0][1], (x[0][0], x[1])))
    
    if cf_type:
        business_user_score_dict_RDD = cleanedRDD.groupByKey().mapValues(dict)
        business_user_score_dict = business_user_score_dict_RDD.collectAsMap()
        business_user_score_dict_broadcast = sc.broadcast(business_user_score_dict)
        # business_list = [i for i in range(num_business)]
        # business_pairs = list(itertools.combinations(business_list, 2))
        # business_pairsRDD = sc.parallelize(business_pairs)
        pairsRDD = sc.parallelize([i for i in range(num_business)]) \
            .flatMap(lambda x: [(x, i) for i in range(x + 1, num_business)])
    else :
        user_business_score_dict_RDD = cleanedRDD.groupByKey().mapValues(dict)
        user_business_score_dict = user_business_score_dict_RDD.collectAsMap()
        user_business_score_dict_broadcast = sc.broadcast(user_business_score_dict)
        pairsRDD = sc.parallelize([i for i in range(num_user)]) \
            .flatMap(lambda x: [(x, i) for i in range(x + 1, num_user)])

    def filter_corated(idx1, idx2):
        if cf_type:
            user_set1 = set(business_user_score_dict_broadcast.value[idx1].keys())
            user_set2 = set(business_user_score_dict_broadcast.value[idx2].keys())
            return len(user_set1 & user_set2) >= corated_threshold
        else:
            business_set1 = set(user_business_score_dict_broadcast.value[idx1].keys())
            business_set2 = set(user_business_score_dict_broadcast.value[idx2].keys())
            common = len(business_set1 & business_set2)
            return common >= corated_threshold and (common / (len(business_set1) + len(business_set2) - common)) >= jaccard_similarity_threshold
    
    def calculate_pearson(idx1, idx2):
        if cf_type:
            dict_1 = business_user_score_dict_broadcast.value[idx1]
            dict_2 = business_user_score_dict_broadcast.value[idx2]
        else:
            dict_1 = user_business_score_dict_broadcast.value[idx1]
            dict_2 = user_business_score_dict_broadcast.value[idx2]
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
        .filter(lambda x: x[2] > 0)
    
    result_pairs = resultRDD.collect()

    print("Total number of pairs in model: {}".format(len(result_pairs)))

    with open(model_file_path, 'w') as f:
        temp = {}
        for one in result_pairs:
            if cf_type:
                temp["b1"] = index_business_dict[one[0]]
                temp["b2"] = index_business_dict[one[1]]
            else:
                temp["u1"] = index_user_dict[one[0]]
                temp["u2"] = index_user_dict[one[1]]
            temp["sim"] = one[2]
            print(json.dumps(temp), file = f)

    end_time = time.time()
    print("Duration: {0:.2f}".format(end_time - start_time))