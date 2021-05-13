import pyspark
import os, sys, time, math, random, json

# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'

random.seed(666)

# hyper-parameter
SAMPLE_SIZE = 2000
LARGE_CLUSTER_PARA = 3
FEW_CLUSTER_THRESHOLD = 10
ALPHA_MAHALANOBIS = 2


def eu_distance(data1: list, data2: list):
    if len(data1) != len(data2):
        raise Exception("Two data points must have same dimension!")
    res = 0
    for i in range(len(data1)):
        res += pow(data1[i] - data2[i], 2)
    return math.sqrt(res)


class KMeans:

    def __init__(self, data_points: dict, num_cluster: int, max_iters=1000):
        '''
        :para data_points: a dict of data points, {index: vector}, vector is a list
        '''
        if num_cluster <= 0:
            raise Exception("num_cluster in k_means has to be positive")

        self.K = num_cluster
        self.max_iters = max_iters
        self.data_points = data_points
        if len(data_points) == 0:
            self.clusters = []
            self.centroids = []
            return

        self._data_points_to_list()

        if len(data_points) <= num_cluster:
            self.clusters = [[i] for i in range(self.num_data)]
            self.centroids = self.X
            return

        # list of list of data index in X for each cluster
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []  # centroid vector of each cluster

        self._predict()

    def get_data_cluster(self):
        '''
        :return: a dict {original data index: cluster index}
        '''
        data_cluster = dict()
        for cluster_idx, cluster in enumerate(self.clusters):
            for data_idx in cluster:
                original_data_idx = self.index_map[data_idx]
                data_cluster[original_data_idx] = cluster_idx
        return data_cluster

    def get_data_split(self, threshold):
        above_data, below_data = [], []
        for cluster in self.clusters:
            if len(cluster) > threshold:
                above_data += list(map(lambda x: self.index_map[x], cluster))
            else:
                below_data += list(map(lambda x: self.index_map[x], cluster))
        return above_data, below_data

    def get_cluster_data(self):
        return list(map(lambda x: list(map(lambda x: self.index_map[x], x)), self.clusters))
        # cluster_data = dict()
        # for cluster_idx, cluster in enumerate(self.clusters):
        #     cluster_data[cluster_idx] = [self.index_map[i] for i in cluster]
        # return cluster_data

    def get_cluster_split(self, threshold):
        # above_clusters = [cluster in self.get_cluster_data() if len(cluster) > threshold]
        # below_clusters = [cluster in self.get_cluster_data() if len(cluster) > threshold]
        above_clusters, below_clusters = [], []
        for cluster in self.get_cluster_data():
            if len(cluster) > threshold:
                above_clusters.append(cluster)
            else:
                below_clusters.append(cluster)
        return above_clusters, below_clusters

    def _data_points_to_list(self):
        self.X = []
        self.index_map = dict()
        for i, (data_index, data_vector) in enumerate(self.data_points.items()):
            self.X.append(data_vector)
            self.index_map[i] = data_index
        self.num_data = len(self.X)
        self.dimension_data = len(self.X[0])

    def _predict(self):
        # initialize centroids
        self._init_centroids()
        # optimization
        for _ in range(self.max_iters):
            # update clusters
            self.clusters = self._update_clusters()
            # update centroids
            centroids_old = self.centroids
            self.centroids = self._update_centroids()
            # check if converged
            if self._is_converged(centroids_old):
                break

    def _init_centroids(self):
        # self.centroids = random.sample(self.X, self.K)
        # # after the first point, pick the rest as far as possible
        self.centroids.append(self.X[0])
        while len(self.centroids) < self.K:
            distances = [min(
                            [eu_distance(data, centroid) for centroid in self.centroids]
                        ) for data in self.X]
            self.centroids.append(self.X[distances.index(max(distances))])

    def _update_clusters(self):
        clusters = [[] for _ in range(self.K)]
        for idx, data in enumerate(self.X):
            centroid_idx = self._find_closest_centroid_idx(data)
            clusters[centroid_idx].append(idx)
        return clusters

    def _update_centroids(self):
        centroids = []
        for cluster in self.clusters:
            sum_vector = [0 for _ in range(self.dimension_data)]
            for idx in cluster:
                sum_vector = [a + b for a, b in zip(sum_vector, self.X[idx])]
            centroids.append([i / len(cluster) for i in sum_vector])
        return centroids

    def _find_closest_centroid_idx(self, data):
        distances = [eu_distance(data, centroid)
                     for centroid in self.centroids]
        closest_idx = distances.index(min(distances))
        return closest_idx

    def _is_converged(self, centroids_old):
        distances = [eu_distance(centroids_old[i], self.centroids[i])
                     for i in range(self.K)]
        return sum(distances) == 0


# data_points = {10: [0,0], 11:[1,1], 12:[2,2], 13:[3,3], 14:[4,4], 15:[1,0], 16:[0,1], 17:[4,3], 18:[3,4]}
# k = KMeans(data_points, 4)
# print(k.get_data_cluster())
# print(k.get_cluster_data())
# print(k.get_cluster_split(2))
# print(k.get_data_split(2))


class DS_CS:  # discard set and compression set share the same class structure
    def __init__(self, data_points: list, data_indices):
        '''
        :para data_points: a list of data vector(list)
        '''
        self.N = len(data_points)
        self.SUM = [0 for _ in range(len(data_points[0]))]
        self.SUMSQ = self.SUM.copy()
        for data_point in data_points:
            self.SUM = [a + b for a, b in zip(self.SUM, data_point)]
            self.SUMSQ = [
                a + b for a, b in zip(self.SUMSQ, list(map(lambda x: x * x, data_point)))]
        self.data_indices = [] + data_indices

    def get_centroid(self):
        return [i / self.N for i in self.SUM]

    def get_variance(self):
        return [(sumsq_i / self.N) - (sum_i / self.N) ** 2 for sumsq_i, sum_i in zip(self.SUMSQ, self.SUM)]

    def add_one_data(self, data_point: [], data_index):
        self.N += 1
        self.SUM = [a + b for a, b in zip(self.SUM, data_point)]
        self.SUMSQ = [
            a + b for a, b in zip(self.SUMSQ, list(map(lambda x: x * x, data_point)))]
        self.data_indices.append(data_index)

    def add_one_DS_CS(self, other):
        self.N += other.N
        for i in range(len(self.SUM)):
            self.SUM[i] += other.SUM[i]
            self.SUMSQ[i] += other.SUMSQ[i]
        self.data_indices += other.data_indices

    def __repr__(self):
        return f"centroid: {str(self.get_centroid())}, variance: {str(self.get_variance())}"


class RS:  # retained set
    def __init__(self, data_points, rs_data_indices):
        self.data_points = {index: data_points[index]
                            for index in rs_data_indices}

    def add_data_points(self, adding_data_points: dict):
        for index, data in adding_data_points.items():
            self.data_points[index] = data


# test = DS_CS([[1,2,3],[2,3,4],[3,4,5]], [0,1,2])
# print(test.get_centroid())
# print(test.get_variance())


def mahalanobis_distance(data_point: list, set_DS_CS: DS_CS):
    centroid = set_DS_CS.get_centroid()
    variance = set_DS_CS.get_variance()
    vector = [((x_i - c_i) / math.sqrt(sigma_i)) ** 2 for x_i, c_i,
              sigma_i in zip(data_point, centroid, variance)]
    return math.sqrt(sum(vector))

# print(mahalanobis_distance([0,0,0], test))


def build_DS_CS(data_points: dict, cluster_data_idx: list):
    return [DS_CS(list(map(lambda x: data_points[x], cluster)), cluster) for cluster in cluster_data_idx]


def assign_to_DS_CS(data_points: dict, DS_CS_list: list):
    if len(DS_CS_list) == 0:
        return data_points
    assigned_data_indices = set()
    for idx, data_point in data_points.items():
        distances = [mahalanobis_distance(data_point, set_DS_CS) for set_DS_CS in DS_CS_list]
        min_distance = min(distances)
        if min_distance > math.sqrt(len(data_point)) * ALPHA_MAHALANOBIS:
            continue
        DS_CS_list[distances.index(min_distance)].add_one_data(data_point, idx)
        assigned_data_indices.add(idx)
    return {k: data_points[k] for k in set(data_points) - assigned_data_indices}


def merge_CS(CS_list: list):
    while len(CS_list) > 1:
        num_dimension = len(CS_list[0].SUM)
        res = []
        for i in range(len(CS_list)):
            for j in range(i + 1, len(CS_list)):
                distance = max(mahalanobis_distance(CS_list[i].get_centroid(), CS_list[j]),
                               mahalanobis_distance(CS_list[j].get_centroid(), CS_list[i]))
                res.append((i, j, distance))
        res.sort(key=lambda x: x[2])
        if res[0][2] > math.sqrt(num_dimension) * ALPHA_MAHALANOBIS:
            break
        CS_list[res[0][0]].add_one_DS_CS(CS_list[res[0][1]])
        CS_list.pop(res[0][1])

    return CS_list


def get_num_points_DS_CS_list(DS_CS_list: list):
    return sum(map(lambda x: x.N, DS_CS_list))


def merge_CS_to_DS(DS_list: list, CS_list: list):
    for one_CS in CS_list:
        distances = [eu_distance(one_CS.get_centroid(), one_DS.get_centroid()) for one_DS in DS_list]
        merge_to_idx = distances.index(min(distances))
        DS_list[merge_to_idx].add_one_DS_CS(one_CS)
    return DS_list


if __name__ == "__main__":

    start_time = time.time()

    # parse commandline argument
    input_path = sys.argv[1]
    n_cluster = int(sys.argv[2])
    output_file_cluster = sys.argv[3]
    output_file_intermediate = sys.argv[4]

    conf = pyspark.SparkConf().setAppName("BFR").setMaster("local[*]")
    sc = pyspark.SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    with open(output_file_intermediate, 'w') as f:
        print("round_id,nof_cluster_discard,nof_point_discard,nof_cluster_compression,nof_point_compression,nof_point_retained", file=f)

    for index, data_chunk_path in enumerate(sorted(os.listdir(input_path))):
        # @7.1 load data
        dataRDD = sc.textFile(input_path + "/" + data_chunk_path).map(lambda x: x.split(",")) \
            .map(lambda x: (x[0], list(map(float, x[1:]))))
        data_points = dataRDD.collectAsMap()

        if index == 0:
            # @7.2 run K-Means on a random sample of the data points
            sampleRDD = dataRDD.sample(
                False, SAMPLE_SIZE / len(data_points), seed=666)
            sample_points = sampleRDD.collectAsMap()
            k_sample = KMeans(sample_points, n_cluster * LARGE_CLUSTER_PARA)
            # @7.3 split data points to outlier and inlier
            inlier_data_idx, outlier_data_idx = k_sample.get_data_split(
                FEW_CLUSTER_THRESHOLD)
            # @7.4 run K-Means on inlier, get K clusters, build DS
            inlier_dataRDD = sc.parallelize(inlier_data_idx).map(
                lambda x: (x, data_points[x]))
            k_inlier = KMeans(inlier_dataRDD.collectAsMap(), n_cluster)
            DS_list = build_DS_CS(data_points, k_inlier.get_cluster_data())
            # @7.5 run K-Means on outlier, build CS and RS
            if len(outlier_data_idx) == 0:
                CS_list = []
                RS_obj = RS(data_points, [])
            else:
                outlier_dataRDD = sc.parallelize(outlier_data_idx).map(
                    lambda x: (x, data_points[x]))
                k_outlier = KMeans(outlier_dataRDD.collectAsMap(),
                                n_cluster * LARGE_CLUSTER_PARA)
                cs_cluster_data, rs_cluster_data = k_outlier.get_cluster_split(1)
                CS_list = build_DS_CS(data_points, cs_cluster_data)
                RS_obj = RS(data_points, [cluster[0]
                            for cluster in rs_cluster_data])
            # delete sample data from data_points
            data_points = {k: data_points[k] for k in set(
                data_points) - set(sample_points)}

        # @7.8 compare new data points to each DS using mahalanobis distance
        data_points = assign_to_DS_CS(data_points, DS_list)
        # @7.9 to each CS
        data_points = assign_to_DS_CS(data_points, CS_list)
        # @7.10 assgin the remaining to RS
        RS_obj.add_data_points(data_points)
        # @7.11 run K-Means on RS, split new CS and RS (similar to @7.5)
        k_RS = KMeans(RS_obj.data_points, n_cluster * LARGE_CLUSTER_PARA)
        cs_cluster_data, rs_cluster_data = k_RS.get_cluster_split(1)
        CS_list += build_DS_CS(RS_obj.data_points, cs_cluster_data)
        RS_obj = RS(RS_obj.data_points, [cluster[0]
                    for cluster in rs_cluster_data])
        # @7.12 merge CS clusters
        CS_list = merge_CS(CS_list)
        # if last iteration, merge all CS to DS
        if index == len(os.listdir(input_path)) - 1:
            DS_list = merge_CS_to_DS(DS_list, CS_list)
            CS_list = []
            RS_obj.data_points = assign_to_DS_CS(RS_obj.data_points, DS_list)

        # @7.13 output intermediate result
        intermediate_result = [index + 1, len(DS_list), get_num_points_DS_CS_list(DS_list),
                               len(CS_list), get_num_points_DS_CS_list(CS_list), len(RS_obj.data_points)]
        with open(output_file_intermediate, 'a') as f:
            print(str(intermediate_result)[1: -1], file=f)

    # output result
    result = {}
    for i, one_DS in enumerate(DS_list):
        for data_index in one_DS.data_indices:
            result[data_index] = i
    for idx in RS_obj.data_points:
        result[idx] = -1

    with open(output_file_cluster, 'w') as f:
        json.dump(result, f)

    print("Duration: {0:.2f}".format(time.time() - start_time))
