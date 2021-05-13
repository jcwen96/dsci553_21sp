import pyspark, pyspark.streaming
import sys, time, json, random, math
import binascii

random.seed(666)

NUM_HASH = 200
LARGEST_HASHCODE = sys.maxsize
NUM_GROUP = 5
GROUP_SIZE = int(NUM_HASH / NUM_GROUP)


def estimate_city(cityRDD):  # using Flajolet-Martin algorithm
    city_set = set(cityRDD.distinct().collect())
    city_set.discard('')
    ground_truth = len(city_set)

    longest_trailing_zeros = [0 for _ in range(NUM_HASH)]
    # 每次都随机生成新的hash function的参数
    hash_para_a = random.sample(range(LARGEST_HASHCODE - 1), NUM_HASH)
    hash_para_b = random.sample(range(LARGEST_HASHCODE - 1), NUM_HASH)

    for city in city_set:
        hashcodes = [(a * int(binascii.hexlify(city.encode("utf8")), 16) - b) % LARGEST_HASHCODE 
                            for a, b in zip(hash_para_a, hash_para_b)]
        hash_str = list(map(lambda x: bin(x), hashcodes))
        trailing_zeros = list(map(lambda x: len(x) - len(x.rstrip("0")), hash_str))
        longest_trailing_zeros = [max(a, b) for a, b in zip(longest_trailing_zeros, trailing_zeros)]
    
    data = sorted([2 ** r for r in longest_trailing_zeros])
    data_group = [data[GROUP_SIZE * i : GROUP_SIZE * (i + 1)] for i in range(NUM_GROUP)]
    avg_group = list(map(lambda x: sum(x) / len(x), data_group))
    median_idx = int(len(avg_group) / 2)
    estimation = round(avg_group[median_idx])

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print()
    print("============DEBUG=============")
    print(city_set)
    print(avg_group)
    print(current_time, ground_truth, estimation, sep=',')
    
    with open(output_file_path, 'a') as f:
        print(current_time, ground_truth, estimation, sep=',', file=f)


if __name__ == "__main__":

    start_time = time.time()

    # parse commandline argument
    port_num = int(sys.argv[1])
    output_file_path = sys.argv[2]

    with open(output_file_path, 'w') as f:
        print("Time,Ground Truth,Estimation", file=f)

    conf = pyspark.SparkConf().setAppName("Task2").setMaster("local[*]")
    sc = pyspark.SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    ssc = pyspark.streaming.StreamingContext(sc, 5) # set the batch duration to 5s

    data_stream = ssc.socketTextStream("localhost", port_num) # connect data stream

    data_stream.window(30, 10).map(lambda x: json.loads(x)).map(lambda x: x["city"]).foreachRDD(estimate_city)

    ssc.start()
    ssc.awaitTermination()

    print("Duration: {0:.2f}".format(time.time() - start_time))