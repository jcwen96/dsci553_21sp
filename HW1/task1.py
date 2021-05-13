import pyspark
import sys, json, operator, re

if __name__ == "__main__":

    # parse commandline argument
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    stopwords_path = sys.argv[3]
    given_year = sys.argv[4]
    top_m_user = int(sys.argv[5])
    top_n_word = int(sys.argv[6])

    conf = pyspark.SparkConf().setAppName("Task1").setMaster("local[*]")
    sc = pyspark.SparkContext(conf = conf)
    sc.setLogLevel("ERROR")

    reviewsRDD = sc.textFile(input_path).map(lambda review: json.loads(review)).cache()
    result = {}

    # total number of reviews
    result["A"] = reviewsRDD.count()
    print("total number of reviews: ", result["A"])

    # total number of reviews in a given year
    result["B"] = reviewsRDD.filter(lambda review: given_year in review["date"]).count()
    print("total number of reviews in a given year: ", result["B"])

    # number of distinct users
    result["C"] = reviewsRDD.map(lambda review: review["user_id"]).distinct().count()
    print("number of distinct users who have written the reviews: ", result["C"])

    # top M users who have the largest number of reviews and its count
    result["D"] = reviewsRDD.map(lambda review: [review["user_id"], 1]).reduceByKey(operator.add).takeOrdered(top_m_user, key=lambda list: [-list[1], list[0]])
    print("top {} users who have the largest number of reviews and its count:".format(top_m_user))
    print(result["D"])

    # top N frequent words in the review text excluding stopwords
    stopwords = sc.textFile(stopwords_path).collect()
    result["E"] = reviewsRDD.map(lambda review: review["text"].translate({ord(i): None for i in ',.!?:;()[]'}).lower()).flatMap(lambda review: review.split()).filter(lambda w: w and w not in stopwords).map(lambda w: (w, 1)).reduceByKey(operator.add).takeOrdered(top_n_word, key=lambda x: (-x[1], x[0]))
    result["E"] = list(map(lambda x: x[0], result["E"]))
    print("top {} frequent words in the review text excluding stopwords:".format(top_n_word))
    print(result["E"])

    # output
    with open(output_path, 'w') as f:
        json.dump(result, f, sort_keys=True)