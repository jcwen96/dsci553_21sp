import pyspark
import sys, itertools, collections, math, operator, time, pickle


def a_priori(all_baskets, support_threshold):
    '''
        @para all_baskets is a list of sets
        @return a dictionary where key is the size of frequent itemsets, value is frequent itemsets
    '''
    C1 = get_single(all_baskets)
    result = dict()
    
    L1 = filter_candidate(all_baskets, C1, support_threshold)
    L_current = L1
    k = 2

    while (L_current):
        result[k - 1] = L_current
        C_current = generate_candidate(L_current, k)
        C_current = pruning(C_current, L_current, k - 1)
        L_current = filter_candidate(all_baskets, C_current, support_threshold)
        k += 1

    return result


def get_single(all_baskets):
    result = set()
    for basket in all_baskets:
        for item in basket:
            result.add(frozenset({item}))
    return result


def generate_candidate(item_sets, length):
    result = set()
    for i in item_sets:
        for j in item_sets:
            temp = i.union(j)
            if (len(temp) == length):
                result.add(temp)
    return result


# @para length: the length of itemset in prev_frequent, not in candidates
def pruning(candidates, prev_frequent, length):
    result = candidates.copy()
    for item in candidates:
        subsets = itertools.combinations(item, length)
        for subset in subsets:
            if (frozenset(subset) not in prev_frequent):
                result.remove(item)
                break
    return result


def filter_candidate(all_baskets, candidates, support_threshold):
    result = set()
    if len(candidates) == 0:
        return result
    
    counts = collections.defaultdict(int)

    for basket in all_baskets:
        for candidate in candidates:
            if candidate.issubset(basket):
                counts[candidate] += 1
    
    for itemset, count in counts.items():
        if count >= support_threshold:
            result.add(itemset)

    return result


# convert the dictionary to a set of all frequent itemsets
def convert_apriori_result(frequent_itemsets_dict):
    result = set()
    for size, itemsets in frequent_itemsets_dict.items():
        for itemset in itemsets:
            result.add(itemset)
    return result


# @para itemsets is a list of itemsets
def write_itemsets(file, itemsets, is_final_res):
    itemsets = list(map(list, itemsets))
    for itemset in itemsets:
        itemset.sort()
    itemsets.sort(key = lambda x: (len(x), x))

    if is_final_res :
        with open("task2_task3_res.tmp", 'wb') as f:
            pickle.dump(itemsets, f)

    k = 1
    start = 0
    for i in range(len(itemsets)):
        if len(itemsets[i]) > k:
            if k == 1:
                file.write(','.join("('{}')".format(x[0]) for x in itemsets[start : i]) + '\n\n')
            else:
                file.write(','.join(str(tuple(x)) for x in itemsets[start : i]) + '\n\n')
            start = i
            k += 1
        i += 1
    if k == 1:
        file.write(','.join("('{}')".format(x[0]) for x in itemsets[start:]) + '\n\n')
    else :
        file.write(','.join(str(tuple(x)) for x in itemsets[start:]) + '\n\n')



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

    def SON_1(iterator):
        chunk_baskets = list(iterator)
        local_threshold = math.ceil((len(chunk_baskets) / num_basket) * support_threshold)
        result = a_priori(chunk_baskets, local_threshold)
        return convert_apriori_result(result)

    # SON phase1
    candidatesRDD = basketsRDD.mapPartitions(SON_1).distinct()
    candidates = candidatesRDD.collect()

    # print candidates
    with open(output_file_path, 'w') as f:
        f.write("Candidates:\n")
        write_itemsets(f, candidates, False)

    def SON_2(iterator):
        counts = collections.defaultdict(int)
        chunk_baskets = list(iterator)

        for basket in chunk_baskets:
            for candidate in candidates:
                if candidate.issubset(basket) :
                    counts[candidate] += 1
        
        return [(itemset, count) for itemset, count in counts.items()]

    # SON phase2
    resultsRDD = basketsRDD.mapPartitions(SON_2) \
        .reduceByKey(operator.add) \
        .filter(lambda x: x[1] >= support_threshold) \
        .map(lambda x: x[0])
    
    result = resultsRDD.collect()

    # print frequent itemsets
    with open(output_file_path, 'a') as f:
        f.write("Frequent Itemsets:\n")
        write_itemsets(f, result, True)

    end_time = time.time()
    print("Duration: {0:.2f}".format(end_time - start_time))
