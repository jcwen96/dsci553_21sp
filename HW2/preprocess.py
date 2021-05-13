import pyspark, pyspark.sql
import sys, json

if __name__ == "__main__":

    business_file_path = sys.argv[1]
    review_file_path = sys.argv[2]

    conf = pyspark.SparkConf().setAppName("PreProcess").setMaster("local[*]")
    sc = pyspark.SparkContext(conf = conf)
    sc.setLogLevel("ERROR")
    spark = pyspark.sql.SparkSession(sc)

    businessRDD = sc.textFile(business_file_path) \
        .map(lambda x: json.loads(x)) \
        .filter(lambda x: x['state'] == 'NV') \
        .map(lambda x: x['business_id'])

    business_NV = set(businessRDD.collect())

    reviewsRDD = sc.textFile(review_file_path) \
        .map(lambda x: json.loads(x)) \
        .filter(lambda x: x['business_id'] in business_NV) \
        .map(lambda x: (x['user_id'], x['business_id']))

    # reviewsRDD.saveAsTextFile("out/preprocess.out")
    df = reviewsRDD.toDF(["user_id", "business_id"])
    df.show()
    print(df.count())
    df.coalesce(1).write.format('com.databricks.spark.csv').options(header='true').save("out/preprocess.csv")
