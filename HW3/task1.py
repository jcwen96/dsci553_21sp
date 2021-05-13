import pyspark
import sys, json, time, itertools


# hyper parameter
similarity_threshold = 0.05
num_hash = 50 # n, length of signature, notice b * r = n
num_bands = 50 # b
num_rows = 1 # r
# hash function: f(x) = (ax + b) % m
hash_para_a = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227]
hash_para_b = [23, 7717, 25837, 18147, 10874, 19457, 3529, 15815, 14757, 11386, 5926, 523, 3395, 19248, 11518, 1957, 24533, 17906, 8190, 17205, 17981, 16428, 2023, 25588, 8933, 17368, 3810, 19747, 23349, 21770, 19924, 1049, 15090, 9201, 15426, 16036, 21421, 23026, 9579, 12832, 14881, 10611, 25773, 5990, 9260, 6784, 18220, 279, 24256, 13808]


if __name__ == "__main__":

    start_time = time.time()

    # parse commandline argument
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    conf = pyspark.SparkConf().setAppName("Task1").setMaster("local[*]")
    sc = pyspark.SparkContext(conf = conf)
    sc.setLogLevel("ERROR")

    rawRDD = sc.textFile(input_file_path) \
        .map(lambda x: json.loads(x)) \
        .map(lambda x: (x["business_id"], x["user_id"]))
    
    userRDD = rawRDD.map(lambda x: x[1]).distinct().zipWithIndex()
    user_dict = userRDD.collectAsMap()
    num_user = len(user_dict)
    user_dict_broadcast = sc.broadcast(user_dict)

    businessRDD = rawRDD.map(lambda x: x[0]).distinct().zipWithIndex()
    business_dict = businessRDD.collectAsMap()
    num_business = len(business_dict)
    business_dict_broadcast = sc.broadcast(business_dict)
    index_business_dict = businessRDD.map(lambda x: (x[1], x[0])).collectAsMap()
    index_business_broadcast = sc.broadcast(index_business_dict)

    rawRDD = rawRDD.map(lambda x: (business_dict_broadcast.value[x[0]],  user_dict_broadcast.value[x[1]]))

    business_user_RDD = rawRDD.groupByKey().mapValues(set)
    business_user_dict = business_user_RDD.collectAsMap()
    business_user_dict_broadcast = sc.broadcast(business_user_dict)

    def min_hash(user_set):
        m = num_user
        index_list = [user_set]
        for i in range(num_hash - 1):
            index_list.append(set(map(lambda x: (hash_para_a[i] * x + hash_para_b[i]) % m, user_set)))
        
        return [min(i) for i in index_list]
    
    def split_signature_into_band(signatrue):
        result = []
        for i in range(num_bands):
            result.append(signatrue[i * num_rows : (i + 1) * num_rows])
        return result

    business_signature_RDD = business_user_RDD \
        .mapValues(lambda x: min_hash(x)) \
        .mapValues(lambda x: split_signature_into_band(x))
    
    business_signature = business_signature_RDD.collectAsMap()
    business_signature_broadcast = sc.broadcast(business_signature)

    # business_list = [i for i in range(num_business)]
    # business_pairs = list(itertools.combinations(business_list, 2))
    # business_pairsRDD = sc.parallelize(business_pairs)
    business_pairsRDD = sc.parallelize([i for i in range(num_business)]) \
            .flatMap(lambda x: [(x, i) for i in range(x + 1, num_business)])

    def filter_pairs(business1, business2):
        signature1 = business_signature_broadcast.value[business1]
        signature2 = business_signature_broadcast.value[business2]
        for i in range(num_bands):
            if signature1[i] == signature2[i]:
                return True
        return False
    
    def compute_jaccard(business1, business2):
        user_set1 = business_user_dict_broadcast.value[business1]
        user_set2 = business_user_dict_broadcast.value[business2]
        common = len(user_set1.intersection(user_set2))
        total = len(user_set1) + len(user_set2) - common
        return common / total

    candidate_RDD = business_pairsRDD.filter(lambda x: filter_pairs(x[0], x[1])) 
    resultRDD = candidate_RDD.map(lambda x: (x[0], x[1], compute_jaccard(x[0], x[1]))) \
        .filter(lambda x: x[2] >= similarity_threshold) \
        .map(lambda x: (index_business_broadcast.value[x[0]], index_business_broadcast.value[x[1]], x[2]))
    
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
