import pyspark
from pyspark.mllib.fpm import FPGrowth
import sys, os, time, pickle


if __name__ == "__main__":

    start_time = time.time()

    # parse commandline argument
    filter_threshold = int(sys.argv[1])
    support_threshold = int(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]

    conf = pyspark.SparkConf().setAppName("Task1").setMaster("local[*]")
    sc = pyspark.SparkContext(conf = conf)
    sc.setLogLevel("ERROR")

    rawRDD = sc.textFile(input_file_path) \
        .filter(lambda x: not x.startswith("user_id")) \
        .map(lambda x: x.split(','))

    basketsRDD = rawRDD.groupByKey() \
        .mapValues(set) \
        .filter(lambda x: len(x[1]) > filter_threshold) \
        .map(lambda x: x[1]).cache()
    
    num_basket = basketsRDD.count()

    model = FPGrowth.train(basketsRDD, support_threshold / num_basket, basketsRDD.getNumPartitions())
    result = model.freqItemsets().collect()

    result = list(map(lambda x: list(x)[0], result))
    for itemset in result:
        itemset.sort()
    result.sort(key = lambda x: (len(x), x))

    result_len = len(result)

    if os.path.exists("task2_task3_res.tmp"):

        with open("task2_task3_res.tmp", "rb") as f:
            task2_res = pickle.load(f)

        with open(output_file_path, 'w') as f:
            print("Task2,{}".format(len(task2_res)), file = f)
            print("Task3,{}".format(result_len), file = f)
            if result == task2_res:
                print("Intersection,{}".format(result_len), file = f)
        
        # os.remove("task2_task3_res.tmp")

    else:
        with open(output_file_path, 'w') as f:
            print("Task2,{}".format(result_len), file = f)
            print("Task3,{}".format(result_len), file = f)
            print("Intersection,{}".format(result_len), file = f)

    end_time = time.time()
    print("Duration: {0:.2f}".format(end_time - start_time))
