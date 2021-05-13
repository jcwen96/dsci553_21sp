import pyspark
import sys, json, operator

if __name__ == "__main__":

    # parse commandline argument
    review_file = sys.argv[1]
    business_file = sys.argv[2]
    output_file = sys.argv[3]
    if_spark = sys.argv[4] # only "spark" and "no_spark"
    if_spark = if_spark == "spark" # no error check for others
    top_n_categories = int(sys.argv[5])

    def filter_categories(categories):
        list = categories.split(",")
        for i in range(len(list)):
            list[i] = list[i].strip()
        return list

    result = {}
    
    if if_spark :
        conf = pyspark.SparkConf().setAppName("Task2").setMaster("local[*]")
        sc = pyspark.SparkContext(conf = conf)
        sc.setLogLevel("ERROR")

        reviewsRDD = sc.textFile(review_file).map(lambda review: json.loads(review)).map(lambda item: (item["business_id"], item["stars"]))
        businessRDD = sc.textFile(business_file).map(lambda item: json.loads(item)).filter(lambda item: item["categories"]).map(lambda item: (item["business_id"], filter_categories(item["categories"])))

        def flat(item):
            list = []
            for c in item[1][1]:
                list.append((c, item[1][0]))
            return list

        reviewsRDD = reviewsRDD.partitionBy(reviewsRDD.getNumPartitions())
        businessRDD = businessRDD.partitionBy(businessRDD.getNumPartitions())
        joinRDD = reviewsRDD.join(businessRDD).flatMap(flat)
        scoreRDD = joinRDD.reduceByKey(operator.add)
        countRDD = joinRDD.map(lambda x: (x[0], 1)).reduceByKey(operator.add)
        avgRDD = scoreRDD.join(countRDD).map(lambda x: (x[0], round(x[1][0] / x[1][1], 1)))

        result["result"] = avgRDD.takeOrdered(top_n_categories, lambda x: (-x[1], x[0]))

    else:
        business_categories = {}
        with open(business_file, 'r', encoding="utf-8") as fp:
            for line in fp:
                line = json.loads(line)
                if line["categories"] == None:
                    continue
                business_categories[line["business_id"]] = filter_categories(line["categories"])

        reviews = []
        with open(review_file, 'r') as fp:
            for line in fp:
                line = json.loads(line)
                reviews.append((line["business_id"], line["stars"]))
        
        category_star = {}
        for review in reviews:
            if review[0] in business_categories:
                for category in business_categories[review[0]]:
                    if category in category_star:
                        category_star[category][0] += review[1]
                        category_star[category][1] += 1
                    else:
                        category_star[category] = [review[1], 1]
        
        for item in category_star.items():
            category_star[item[0]] = round(item[1][0] / item[1][1], 1)

        result["result"] = []
        count = 0
        for k, v in sorted(category_star.items(), key=lambda item: (-item[1], item[0])):
            result["result"].append((k, v))
            count += 1
            if count == top_n_categories:
                break

    print(result)
    with open(output_file, 'w') as f:
        json.dump(result, f, sort_keys=True)
        