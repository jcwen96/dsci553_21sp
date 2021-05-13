import pyspark
import sys, json, time

# hyper parameter
num_neighbor = 5 # for item-based


if __name__ == "__main__":

    start_time = time.time()

    # parse commandline argument
    train_file_path = sys.argv[1]
    test_file_path = sys.argv[2]
    model_file_path = sys.argv[3] # input
    output_file_path = sys.argv[4]
    cf_type = sys.argv[5]
    cf_type = cf_type == "item_based"

    conf = pyspark.SparkConf().setAppName("Task3Predict").setMaster("local[*]")
    sc = pyspark.SparkContext(conf = conf)
    sc.setLogLevel("ERROR")

    # force crush if no this file
    if cf_type:
        avg_dict = json.load(open("data/business_avg.json"))
    else:
        avg_dict = json.load(open("data/user_avg.json"))
    avg_broadcast = sc.broadcast(avg_dict)

    rawRDD = sc.textFile(train_file_path).map(lambda x: json.loads(x)) \
        .map(lambda x: (x["business_id"], x["user_id"], x["stars"]))

    userRDD = rawRDD.map(lambda x: x[1]).distinct().zipWithIndex()
    user_dict = userRDD.collectAsMap()
    user_dict_broadcast = sc.broadcast(user_dict)
    index_user_dict = userRDD.map(lambda x: (x[1], x[0])).collectAsMap()
    index_user_dict_broadcast = sc.broadcast(index_user_dict)
    businessRDD = rawRDD.map(lambda x: x[0]).distinct().zipWithIndex()
    business_dict = businessRDD.collectAsMap()
    business_dict_broadcast = sc.broadcast(business_dict)

    # cleanedRDD: ((user_idx, business_idx), stars)
    cleanedRDD = rawRDD.map(lambda x: (x[0], (x[1], x[2]))) \
        .join(businessRDD).map(lambda x: (x[1][0][0], (x[1][1], x[1][0][1]))) \
        .join(userRDD).map(lambda x: ((x[1][1], x[1][0][0]), x[1][0][1])) \
        .groupByKey().mapValues(list).mapValues(lambda x: sum(x) / len(x))
    
    user_business_star = cleanedRDD.collectAsMap()
    user_business_star_broadcast = sc.broadcast(user_business_star)

    if cf_type:
        user_business_RDD = cleanedRDD.map(lambda x: x[0]).groupByKey().mapValues(set)
        user_business_dict = user_business_RDD.collectAsMap()
        user_business_dict_broadcast = sc.broadcast(user_business_dict)
    else:
        business_user_RDD = cleanedRDD.map(lambda x: ((x[0][1], x[0][0]), x[1])) \
            .map(lambda x: x[0]).groupByKey().mapValues(set)
        business_user_dict = business_user_RDD.collectAsMap()
        business_user_dict_broadcast = sc.broadcast(business_user_dict)

    modelRDD = sc.textFile(model_file_path).map(lambda x: json.loads(x))
    if cf_type:
        modelRDD = modelRDD.map(lambda x: (x['b1'], (x['b2'], x['sim']))) \
            .join(businessRDD).map(lambda x: (x[1][0][0], (x[1][1], x[1][0][1]))) \
            .join(businessRDD).map(lambda x: (frozenset((x[1][0][0], x[1][1])), x[1][0][1]))
    else:
        modelRDD = modelRDD.map(lambda x: (x['u1'], (x['u2'], x['sim']))) \
            .join(userRDD).map(lambda x: (x[1][0][0], (x[1][1], x[1][0][1]))) \
            .join(userRDD).map(lambda x: (frozenset((x[1][0][0], x[1][1])), x[1][0][1]))
    
    model_dict = modelRDD.collectAsMap()
    model_dict_broadcast = sc.broadcast(model_dict)

    def predict(pair_dict):
        user_id = pair_dict['user_id']
        business_id = pair_dict['business_id']
        # cold start for new user or business
        if (user_id not in user_dict_broadcast.value) or \
            (business_id not in business_dict_broadcast.value):
            pair_dict['stars'] = avg_broadcast.value["UNK"]
            return pair_dict
        user_id_idx = user_dict_broadcast.value[user_id]
        business_id_idx = business_dict_broadcast.value[business_id]

        if cf_type:
            neighbor_set = user_business_dict_broadcast.value[user_id_idx]
        else:
            neighbor_set = business_user_dict_broadcast.value[business_id_idx]
        neighbor_model_list = []
        for one in neighbor_set:
            if cf_type:
                pair = frozenset((one, business_id_idx))
            else:
                pair = frozenset((one, user_id_idx))
            if pair in model_dict_broadcast.value:
                neighbor_model_list.append((one, model_dict_broadcast.value[pair]))
        # no neighbor
        if len(neighbor_model_list) == 0:
            if cf_type:
                pair_dict['stars'] = avg_broadcast.value[business_id]
            else:
                pair_dict['stars'] = avg_broadcast.value[user_id]
            return pair_dict
        
        neighbor_model_list.sort(key = lambda x: x[1], reverse=True)
        neighbor_model_list = neighbor_model_list[:num_neighbor]

        numerator, denominator = 0, 0
        for one in neighbor_model_list:
            if cf_type:
                star = user_business_star_broadcast.value[(user_id_idx, one[0])]
            else:
                star = user_business_star_broadcast.value[(one[0], business_id_idx)] - avg_broadcast.value[index_user_dict_broadcast.value[one[0]]]
            numerator += star * one[1]
            denominator += abs(one[1])
        
        # if denominator == 0:
        #     pair_dict['stars'] = business_avg_broadcast.value[business_id]
        #     return pair_dict
        
        if cf_type:
            pair_dict['stars'] = numerator / denominator
        else:
            pair_dict['stars'] = (numerator / denominator) + avg_broadcast.value[user_id]
        return pair_dict
    
    # pair = json.loads('{"user_id": "uocYGE8tosU7caXmZl3sxw", "business_id": "G-5kEa6E6PD5fkBRuA7k9Q"}')
    # predict(pair)


    test_predict_RDD = sc.textFile(test_file_path).map(lambda x: json.loads(x)) \
        .map(predict)
    
    predicts = test_predict_RDD.collect()
    print("Number of the whole pairs: {}".format(len(predicts)))
    
    with open(output_file_path, 'w') as f:
        for one in predicts:
            print(json.dumps(one), file=f)
    

    end_time = time.time()
    print("Duration: {0:.2f}".format(end_time - start_time))