import pyspark, pyspark.sql, graphframes
import os, sys, time
from graphframes.examples import Graphs

# os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")

if __name__ == "__main__":

    start_time = time.time()

    # parse commandline argument
    filter_threshold = int(sys.argv[1])
    input_file_path = sys.argv[2]
    community_output_file_path = sys.argv[3]

    conf = pyspark.SparkConf().setAppName("Task1").setMaster("local[*]")
    sc = pyspark.SparkContext(conf = conf)
    sc.setLogLevel("ERROR")

    sqlContext = pyspark.sql.SparkSession.builder \
        .appName("Task1").master("local[*]").getOrCreate()
    
    rawDF = sqlContext.read \
        .option("header", "true") \
        .option("inferSchema", value = True) \
        .csv(input_file_path)
    rawRDD = rawDF.rdd
    
    userRDD = rawRDD.map(lambda x: x[0]).distinct().zipWithIndex()
    user_idx_dict = userRDD.collectAsMap()
    num_user = userRDD.count()
    idx_user_dict = userRDD.map(lambda x: (x[1], x[0])).collectAsMap()

    bizRDD = rawRDD.map(lambda x: x[1]).distinct().zipWithIndex()
    biz_idx_dict = bizRDD.collectAsMap()

    cleanedRDD = rawRDD.map(lambda x: (user_idx_dict[x[0]], biz_idx_dict[x[1]]))

    user_bizRDD = cleanedRDD.groupByKey().mapValues(set)
    user_biz_dict = user_bizRDD.collectAsMap()

    user_pairsRDD = sc.parallelize([i for i in range(num_user)]) \
        .flatMap(lambda x: [(x, i) for i in range(x + 1, num_user)])
    
    def filter_edge(user_pair):
        biz1 = user_biz_dict[user_pair[0]]
        biz2 = user_biz_dict[user_pair[1]]
        return len(biz1 & biz2) >= filter_threshold

    edgeRDD = user_pairsRDD.filter(filter_edge)
    nodeRDD = edgeRDD.flatMap(lambda x: x).distinct()
    # undirected graph, edges need to be both way
    edgeRDD = edgeRDD.flatMap(lambda x: [x, (x[1], x[0])]) \
        .map(lambda x: (idx_user_dict[x[0]], idx_user_dict[x[1]]))
    edgeDF = edgeRDD.toDF(["src", "dst"])
    print("number of edges in the graph:", edgeDF.count())
    nodeDF = nodeRDD.map(lambda x: (idx_user_dict[x],)).toDF(["id"])
    print("number of nodes in the graph:", nodeDF.count())

    # build graph的node和edge还是要用原本的user_id，坑。。。
    g = graphframes.GraphFrame(nodeDF, edgeDF)
    resultDF = g.labelPropagation(maxIter=5)

    # (label, list of user_id)
    resultRDD = resultDF.rdd.map(lambda x: (x[1], x[0])) \
        .groupByKey().mapValues(list) \
        .map(lambda x: sorted(x[1])) \
        .sortBy(lambda x: (len(x), x[0]))
    
    result = resultRDD.collect()

    with open(community_output_file_path, 'w') as f:
        for one in result:
            print(str(one)[1: -1], file = f)

    end_time = time.time()
    print("Duration: {0:.2f}".format(end_time - start_time))