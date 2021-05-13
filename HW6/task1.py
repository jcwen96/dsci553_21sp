import pyspark
import sys, time, json
import binascii

# hyper parameter for hash function
LENGTH_BIT_ARRAY = 10000
# NUM_HASH = 8 # optimal k = (n / m) * ln2
HASH_PARA_A = [1, 2, 3, 5, 7, 11, 13, 17]
HASH_PARA_B = [23, 7717, 5837, 8147, 874, 457, 3529, 15]


if __name__ == "__main__":

    start_time = time.time()

    # parse commandline argument
    first_json_path = sys.argv[1]
    second_json_path = sys.argv[2]
    output_file_path = sys.argv[3]

    conf = pyspark.SparkConf().setAppName("Task1").setMaster("local[*]")
    sc = pyspark.SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    cityRDD = sc.textFile(first_json_path).map(lambda x: json.loads(x)).map(lambda x: x["city"])
    city_set = set(cityRDD.distinct().collect())
    city_set.discard('')
    
    bit_array = [0 for _ in range(LENGTH_BIT_ARRAY)]

    for city in city_set:
        hashcodes = [(a * int(binascii.hexlify(city.encode("utf8")), 16) + b) % LENGTH_BIT_ARRAY for a, b in zip(HASH_PARA_A, HASH_PARA_B)]
        for hashcode in hashcodes:
            bit_array[hashcode] = 1
    
    predictRDD = sc.textFile(second_json_path).map(lambda x: json.loads(x)).map(lambda x: x["city"])
    predict_result = []

    for city in predictRDD.collect():
        if city:
            hashcodes = [(a * int(binascii.hexlify(city.encode("utf8")), 16) + b) % LENGTH_BIT_ARRAY for a, b in zip(HASH_PARA_A, HASH_PARA_B)]
            hash_res = [bit_array[hashcode] for hashcode in hashcodes]
            if 0 in hash_res:
                predict_result.append(0)
            else:
                predict_result.append(1)
        else:
            predict_result.append(0) # 默认city为空时，设为0
    
    with open(output_file_path, 'w') as f:
        print(*predict_result, file = f)

    
    print("Duration: {0:.2f}".format(time.time() - start_time))