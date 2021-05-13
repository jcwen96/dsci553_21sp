import pyspark
import sys, os, time, pickle
from collections import defaultdict

class TreeNode:
    def __init__(self, name, freq, parent):
        self.name = name
        self.count = freq
        self.parent = parent
        self.children = {}
        self.next = None # LinkedList


def fp_growth(baskets, support_threshold):
    '''
    @para baskets is a list of sets
    '''
    # get single frequency
    freq = [1 for i in range(len(baskets))]
    fp_tree, header_table = construct_tree(baskets, freq, support_threshold)
    result = []
    mine_tree(header_table, support_threshold, set(), result)
    return result

def construct_tree(baskets, freq, support_threshold):
    '''
    PRE: len(baskets) == len(freq)
    '''
    header_table = defaultdict(int)
    for i, basket in enumerate(baskets):
        for item in basket:
            header_table[item] += freq[i]
    
    header_table = dict((item, sup) for item, sup in header_table.items() if sup >= support_threshold)
    if (len(header_table) == 0):
        return None, None
    
    # add one more field in header_table: {item: [support, LinkedList_header]}
    for item in header_table:
        header_table[item] = [header_table[item], None]

    fp_tree = TreeNode('Null', 1, None)
    for idx, basket in enumerate(baskets):
        basket = [item for item in basket if item in header_table]
        basket.sort(key = lambda item: header_table[item][0], reverse = True)
        cur_node = fp_tree
        for item in basket:
            cur_node = update_tree(item, cur_node, header_table, freq[idx])
    
    return fp_tree, header_table

def update_tree(item, tree_node, header_table, count):
    if item in tree_node.children:
        tree_node.children[item].count += count
    else:
        new_node = TreeNode(item, count, tree_node)
        tree_node.children[item] = new_node
        update_header_table(item, new_node, header_table)
    
    return tree_node.children[item]

def update_header_table(item, new_node, header_table):
    # LL insert head
    new_node.next = header_table[item][1]
    header_table[item][1] = new_node

def mine_tree(header_table, support_threshold, prefix, result):
    '''
    recersive mine the tree
    '''
    sorted_single_list = [item[0] for item in sorted(list(header_table.items()), key = lambda x: x[1][0])]
    for item in sorted_single_list:
        # pattern growth
        new_freq_set = prefix.copy()
        new_freq_set.add(item)
        result.append(new_freq_set)

        conditional_pattern_base, freq = find_prefix_path(item, header_table)
        conditional_tree, new_header_table = construct_tree(conditional_pattern_base, freq, support_threshold)

        if new_header_table != None:
            mine_tree(new_header_table, support_threshold, new_freq_set, result)

def find_prefix_path(base_patthern, header_table):
    cur_node = header_table[base_patthern][1]
    conditional_patterns = []
    frequency = []
    while cur_node != None:
        prefix_path = []
        get_path(cur_node, prefix_path)
        if len(prefix_path) > 1:
            conditional_patterns.append(prefix_path[1:])
            frequency.append(cur_node.count)
        
        cur_node = cur_node.next
    
    return conditional_patterns, frequency

def get_path(cur_node, prefix_path):
    if cur_node.parent != None:
        prefix_path.append(cur_node.name)
        get_path(cur_node.parent, prefix_path)
    


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

    result = fp_growth(basketsRDD.collect(), support_threshold)

    result = list(map(list, result))
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
