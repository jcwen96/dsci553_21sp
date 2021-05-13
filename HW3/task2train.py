import pyspark
import sys, json, time, collections, operator, math, pickle

# hyper-parameter
rareword_threshold = 0.000001


if __name__ == "__main__":

    start_time = time.time()

    # parse commandline argument
    train_file_path = sys.argv[1] # input
    model_file_path = sys.argv[2] # output
    stopwords_path = sys.argv[3]

    conf = pyspark.SparkConf().setAppName("Task2Train").setMaster("local[*]")
    sc = pyspark.SparkContext(conf = conf)
    sc.setLogLevel("ERROR")

    stopwords = set(sc.textFile(stopwords_path).collect())
    stopwords_broadcast = sc.broadcast(stopwords)

    def only_keep_letter_space(str):
        whitelist = set('abcdefghijklmnopqrstuvwxyz ')
        return "".join(filter(whitelist.__contains__, str.lower()))

    rawRDD = sc.textFile(train_file_path) \
        .map(lambda x: json.loads(x)) \
        .map(lambda x: ((x["user_id"], x["business_id"]), x["text"])) \
        .mapValues(lambda x: only_keep_letter_space(x)) \
        .mapValues(lambda x: list(filter(lambda x: x not in stopwords_broadcast.value, x.split())))
    
    all_wordsRDD = rawRDD.flatMap(lambda x: x[1]).map(lambda x: (x, 1))
    num_total_words = all_wordsRDD.count()
    commonwordsRDD = all_wordsRDD.reduceByKey(operator.add) \
        .filter(lambda x: x[1] > num_total_words * rareword_threshold) \
        .map(lambda x: x[0]).zipWithIndex()
    commonwords = commonwordsRDD.collectAsMap()
    commonwords_broadcast = sc.broadcast(commonwords)

    rawRDD = rawRDD \
        .mapValues(lambda x: list(filter(lambda x: x in commonwords_broadcast.value, x))) \
        .mapValues(lambda x: list(map(lambda x: commonwords_broadcast.value[x], x)))

    def count_word_frequency(lists):
        counts = collections.defaultdict(int)
        for one_list in lists:
            for word in one_list:
                counts[word] += 1
        return counts
        
    business_profile_RDD = rawRDD.map(lambda x: (x[0][1], x[1])) \
        .groupByKey().mapValues(list) \
        .mapValues(count_word_frequency)

    idf_num_business = business_profile_RDD.count()
    idf = business_profile_RDD.mapValues(lambda x: set(x.keys())) \
        .flatMap(lambda x: x[1]).map(lambda x: (x, 1)) \
        .reduceByKey(operator.add).mapValues(lambda x: math.log2(idf_num_business / x)) \
        .collectAsMap()
    idf_broadcast = sc.broadcast(idf)

    def calculate_tf_idf(worddict):
        max_frequency = max(worddict.values())
        for word, frequency in worddict.items():
            worddict[word] = (frequency / max_frequency) * idf_broadcast.value[word]
        return worddict

    business_profile_RDD = business_profile_RDD.mapValues(calculate_tf_idf) \
        .mapValues(lambda x: sorted(x.items(), key=lambda x: x[1], reverse=True)[:200]) \
        .mapValues(lambda x: set(map(lambda x: x[0], x)))
    business_profile = business_profile_RDD.collectAsMap()
    business_profile_broadcast = sc.broadcast(business_profile)

    def build_user_profile(business_list):
        res = set()
        for business in business_list:
            res = res.union(business_profile_broadcast.value[business])
        return res

    # user profile 只是简单的 boolean vector，此处可以优化
    # 可以参考 slide part1 p40 优化 aggregate
    user_profile_RDD = rawRDD.map(lambda x: (x[0][0], x[0][1])) \
        .groupByKey().mapValues(set) \
        .mapValues(build_user_profile)
    
    user_profile = user_profile_RDD.collectAsMap()

    with open(model_file_path, 'wb') as f:
        pickle.dump([business_profile, user_profile], f)
    
    end_time = time.time()
    print("Duration: {0:.2f}".format(end_time - start_time))