import pyspark
import sys, json, time, pickle, math

# hyper-parameter
sim_threshold = 0.01


if __name__ == "__main__":

    start_time = time.time()

    # parse commandline argument
    test_file_path = sys.argv[1] # input
    model_file_path = sys.argv[2] # output
    output_file_path = sys.argv[3]

    conf = pyspark.SparkConf().setAppName("Task2Predict").setMaster("local[*]")
    sc = pyspark.SparkContext(conf = conf)
    sc.setLogLevel("ERROR")

    rawRDD = sc.textFile(test_file_path).map(lambda x: json.loads(x))

    with open(model_file_path, 'rb') as f:
        business_profile, user_profile = pickle.load(f)
    
    business_profile_broadcast = sc.broadcast(business_profile)
    user_profile_broadcast = sc.broadcast(user_profile)

    def compute_cosine(pair_dict):
        if pair_dict["user_id"] not in user_profile_broadcast.value or pair_dict["business_id"] not in business_profile_broadcast.value:
            pair_dict["sim"] = 0
            return pair_dict
        user = user_profile_broadcast.value[pair_dict["user_id"]]
        business = business_profile_broadcast.value[pair_dict["business_id"]]
        dot_product = len(user.intersection(business))
        euclidean_dis = math.sqrt(len(user)) * math.sqrt(len(business))
        pair_dict["sim"] = dot_product / euclidean_dis
        return pair_dict

    res_pairsRDD = rawRDD.map(compute_cosine).filter(lambda x: x["sim"] >= sim_threshold)
    res_pairs = res_pairsRDD.collect()
    print("Number of valid pairs: {}".format(len(res_pairs)))

    business_profile_broadcast.destroy()
    user_profile_broadcast.destroy()
    del business_profile
    del user_profile

    with open(output_file_path, 'w') as f:
        for one in res_pairs:
            print(json.dumps(one), file=f)
    
    end_time = time.time()
    print("Duration: {0:.2f}".format(end_time - start_time))