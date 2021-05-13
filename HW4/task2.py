import pyspark
import os, sys, time
from collections import defaultdict


class TreeNode:
    def __init__(self, name):
        self.name = name
        self.level = 0
        self.parent = {} # {node_name: node}
        self.children = {} # {node_name: node}
        self.node_score = 1
        self.edge_score = 0

    def __repr__(self):
        return f"[name: {self.name}, level: {self.level}, parent: {self.parent.keys()}, children: {self.children.keys()}]"
    


def find_largest_M_communities(graph, edges, node_degree):
    '''
    
    :para graph: adjacentcy list {node: a set of nodes}
    :para edges: a set of all edges in frozenset of the original graph
    :para node_degree: {node: degree(int)} of original graph (for ki and kj)
    :return: a list of communites with max modularity
    '''
    max_modularity, max_communities = calcu_modularity(graph, edges, node_degree)
    left_edges = edges.copy()
    while (len(left_edges) > 0):
        edge_between = calcu_between(graph)
        edges_to_remove = take_highest_between(edge_between)
        graph = remove_edges(graph, edges_to_remove)
        for edge in edges_to_remove:
            left_edges.remove(edge)
        cur_modularity, cur_communities = calcu_modularity(graph, edges, node_degree)
        if cur_modularity > max_modularity:
            max_modularity, max_communities = cur_modularity, cur_communities
    
    return max_communities


def calcu_modularity(graph, edges, node_degree):
    '''
    
    :para graph: adjacentcy list {node: a set of nodes}
    :para edges: a set of all edges in frozenset of the original graph
    :para node_degree: {node: degree(int)} of the original graph (for ki and kj)
    '''
    m_double = sum(node_degree.values()) # 2m, i.e. doulbe of num edges in original graph
    res = 0
    communities = find_communities(graph)
    for community in communities:
        for node_i in community:
            k_i = node_degree[node_i]
            for node_j in community:
                k_j = node_degree[node_j]
                res -= k_i * k_j / m_double
                if frozenset({node_i, node_j}) in edges:
                    res += 1
    return (res / m_double), communities


def find_communities(graph):
    '''
    :return: a list of communities, each community is a set
    '''
    communities = []
    node_visited = set()
    for node in graph.keys():
        if node not in node_visited:
            community = find_community_bfs(graph, node)
            communities.append(community)
            node_visited = node_visited.union(community)
    return communities


def find_community_bfs(graph, node):
    '''
    :return: a set of nodes
    '''
    visited = {node}
    queue = [node]
    while queue:
        node = queue.pop(0)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited


def take_highest_between(edge_between):
    '''
    :para edge_between: {frozenset: float}
    :return: a set of edges with highest between {frozenset()}
    '''
    if len(edge_between) < 2:
        return set(edge_between.keys())
    sorted_edge_between = sorted(list(edge_between.items()), key = lambda x: -x[1])
    res = {sorted_edge_between[0][0]}
    max_between = sorted_edge_between[0][1]
    for one in sorted_edge_between:
        if one[1] == max_between:
            res.add(one[0])
        else:
            break

    return res


def remove_edges(graph, edges_to_remove):
    '''
    :para graph: adjacentcy list {node: a set of nodes}
    :para edges_to_remove: a set of edges in frozenset
    '''
    for edge in edges_to_remove:
        node1, node2 = list(edge)[0], list(edge)[1]
        graph[node1].remove(node2)
        graph[node2].remove(node1)
    return graph


def calcu_between(graph: dict):
    # the result is a dict {frozenset:float}
    res = defaultdict(float)
    for node_name in graph.keys():
        bfs_tree = build_bfs_tree(graph, node_name)
        height = max(bfs_tree.keys())
        for level in range(height, 0, -1):
            for node in bfs_tree[level]:
                cur_score_sum = 1 + node.edge_score
                for parent_name, parent_node in node.parent.items():
                    edge_between = (cur_score_sum / node.node_score) * parent_node.node_score
                    parent_node.edge_score += edge_between
                    edge = frozenset({node.name, parent_name})
                    res[edge] += edge_between
    for k in res:
        res[k] /= 2
    return res


def build_bfs_tree(graph: dict, root_name):
    root = TreeNode(root_name)
    # 返回的tree用{level: node}表示，方便之后bottom up算betweenness
    tree = defaultdict(set)
    tree[0].add(root)
    created = {root_name: root}
    queue = [root_name] # 这个都是node的index，不是treenode本身
    while queue:
        cur_name = queue.pop(0)
        for neighbor_name in graph[cur_name]:
            if neighbor_name not in created.keys():
                new_node = TreeNode(neighbor_name) # create a new node
                new_node.level = 1 + created[cur_name].level # set the level
                new_node.parent[cur_name] = created[cur_name] # set the parent of the new node
                new_node.node_score = created[cur_name].node_score # set the node score of the new node
                tree[new_node.level].add(new_node) # add to the return map
                created[cur_name].children[neighbor_name] = new_node # add it to parent children
                created[neighbor_name] = new_node # add to created
                queue.append(neighbor_name)
            else:
                neighbor_node = created[neighbor_name]
                if neighbor_node.level > created[cur_name].level:
                    neighbor_node.node_score += created[cur_name].node_score
                    created[cur_name].children[neighbor_name] = neighbor_node
                    neighbor_node.parent[cur_name] = created[cur_name]
    return tree



# graph = {'A': {'B', 'D'}, 'B': {'A', 'C', 'E'}, 'C': {'B', 'F'}, 'D': {'A', 'E'}, 'E': {'D', 'B', 'F'}, 'F': {'C', 'E'}}
# # # graph = {'A': {'B', 'C'}, 'B': {'A', 'C'}, 'C': {'B', 'A'}, 'D': {'G', 'F', 'E'}, 'E': {'D', 'F'}, 'F': {'D', 'E', 'G'}, 'G': {'D', 'F'}}
# # # build_bfs_tree(graph, 'A')
# # # build_bfs_tree(graph, 'E')
# node_degree = dict()
# for node, neighbors in graph.items():
#     node_degree[node] = len(neighbors)

# edge_between = calcu_between(graph)
# res = find_largest_M_communities(graph, set(edge_between.keys()), node_degree)

# # edges_to_remove = take_highest_between(edge_between)
# # graph = remove_edges(graph, edges_to_remove)

if __name__ == "__main__":

    start_time = time.time()

    # parse commandline argument
    filter_threshold = int(sys.argv[1])
    input_file_path = sys.argv[2]
    betweenness_output_file_path = sys.argv[3]
    community_output_file_path = sys.argv[4]

    conf = pyspark.SparkConf().setAppName("Task1").setMaster("local[*]")
    sc = pyspark.SparkContext(conf = conf)
    sc.setLogLevel("ERROR")

    rawRDD = sc.textFile(input_file_path) \
        .filter(lambda x: not x.startswith("user_id")) \
        .map(lambda x: x.split(','))

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

    # undirected graph, edges need to be both way
    edgeRDD = user_pairsRDD.filter(filter_edge).flatMap(lambda x: [x, (x[1], x[0])])
    # nodeRDD = edgeRDD.flatMap(lambda x: x).distinct()
    # use adjacency list to represent the graph
    graphRDD = edgeRDD.groupByKey().mapValues(set)
    graph = graphRDD.collectAsMap()

    # for ki and kj
    node_degree = dict()
    for node, neighbors in graph.items():
        node_degree[node] = len(neighbors)

    # {frozenset: float} {edge: betweenness}
    edgeidx_between = calcu_between(graph)

    # task 2.1
    # change to {tuple: float} and change back to original user_id
    edge_between = {}
    for edge_idx, between in edgeidx_between.items():
        edge_between[tuple(sorted(map(lambda x: idx_user_dict[x], edge_idx)))] = between
    
    with open(betweenness_output_file_path, 'w') as f:
        for one in sorted(list(edge_between.items()), key = lambda x: (-x[1], x[0])):
            print(str(one)[1: -1], file = f)
    

    # task 2.2
    communities_largest_m = find_largest_M_communities(graph, set(edgeidx_between.keys()), node_degree)
    # change back to origianl user_id
    res = list(map(lambda x : sorted(map(lambda x: idx_user_dict[x], x)), communities_largest_m))
    res = list(map(lambda x: (x, len(x)), res))
    with open(community_output_file_path, 'w') as f:
        for one in sorted(res, key = lambda x: (x[1], x[0])):
            print(str(one[0])[1: -1], file = f)
    

    end_time = time.time()
    print("Duration: {0:.2f}".format(end_time - start_time))