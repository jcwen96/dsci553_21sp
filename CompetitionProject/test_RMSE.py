import pyspark
import os, sys, json, time, math

os.environ['PYSPARK_PYTHON'] = 'python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python3'

if __name__ == "__main__":

    start_time = time.time()

    test_review_file_path = "data/test_review_ratings.json"
    predict_file_path = "out/predict.out"

    conf = pyspark.SparkConf().setAppName("FinalProjectPredict").setMaster("local[*]")
    sc = pyspark.SparkContext(conf = conf)
    sc.setLogLevel("ERROR")

    ground_truth_RDD = sc.textFile(test_review_file_path).map(lambda x: json.loads(x)) \
        .map(lambda x: ((x["user_id"], x["business_id"]), x["stars"]))
    ground_truth_dict = ground_truth_RDD.collectAsMap()

    predict_RDD = sc.textFile(predict_file_path).map(lambda x: json.loads(x)) \
        .map(lambda x: ((x["user_id"], x["business_id"]), x["stars"]))
    predict_dict = predict_RDD.collectAsMap()

    SUMSQ = 0.0
    count = 0

    error_counts = [0 for i in range(6)]
    for key, value in ground_truth_dict.items():
        if key in predict_dict:
            error = abs(value - predict_dict[key])
            SUMSQ += error ** 2
            count += 1
            error_counts[math.ceil(error)] += 1
    
    RMSE = math.sqrt(SUMSQ / count)
    print("count:", count, "/", len(ground_truth_dict))
    print("errors", error_counts)
    print("RMSE:", RMSE)


    print("Duration: {0:.2f}".format(time.time() - start_time))