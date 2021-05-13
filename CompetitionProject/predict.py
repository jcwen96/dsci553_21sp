import pyspark
import os, sys, json, time

import pandas
from surprise import Dataset
from surprise import Reader
from surprise import SVD
# from surprise.model_selection import GridSearchCV

os.environ['PYSPARK_PYTHON'] = 'python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = 'python3'

# hyper parameter
# num_neighbor = 5 # for item-based
# sim_threshold = 0.3
# neighbor_threshold = 3


if __name__ == "__main__":

    start_time = time.time()

    # parse commandline argument
    train_file_path = "data/train_review.json"
    # model_file_path = "out/model.out" # input
    test_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    conf = pyspark.SparkConf().setAppName("FinalProjectPredict").setMaster("local[*]")
    sc = pyspark.SparkContext(conf = conf)
    sc.setLogLevel("ERROR")

    train_RDD = sc.textFile(train_file_path).map(lambda x: json.loads(x)) \
        .map(lambda x: (x["user_id"], x["business_id"], x["stars"]))

    test_RDD = sc.textFile(test_file_path).map(lambda x: json.loads(x)) \
        .map(lambda x: (x["user_id"], x["business_id"], 0.0))

    train_DF = pandas.DataFrame(   {"user_id": train_RDD.map(lambda x: x[0]).collect(), 
                                    "business_id": train_RDD.map(lambda x: x[1]).collect(), 
                                    "rating": train_RDD.map(lambda x: x[2]).collect()}  )


    reader = Reader(rating_scale=(1, 5))
    train_data = Dataset.load_from_df(train_DF, reader)
    train_set = train_data.build_full_trainset()

    algo = SVD(n_epochs = 27, lr_all = 0.005, reg_all = 0.1) # parameter selection done by 10 fold cross validation
    algo.fit(train_set)

    test_set = test_RDD.collect()

    # # param_grid = {'n_epochs': [20, 27], 'lr_all': [0.002, 0.005],
    # #                 'reg_all': [0.1, 0.2]}
    # # gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=10)
    # # gs.fit(train_data)

    # # print(gs.best_score['rmse'])
    # # print(gs.best_params['rmse'])


    # test = algo.predict("1JEXL5K6VTx01tAs6Jskkg", "M30I1NPl5JuHthxo1IXPGg", verbose=True)
    # print(test[3])
    

    predictions = algo.test(test_set)

    result = list(map(lambda x: {"user_id": x.uid, "business_id": x.iid, "stars": x.est}, predictions))
    

    with open(output_file_path, 'w') as f:
        for one in result:
            print(json.dumps(one), file=f)

    print("Number of the prediction pairs: {}".format(len(predictions)))
    print("Duration: {0:.2f}".format(time.time() - start_time))

    # # force crush if no this file
    # # avg_dict = json.load(open("data/business_avg.json"))
    # # avg_backup_dict = json.load(open("out/business_avg.json"))

    # # x["business_id"], x["user_id"], x["stars"]
    # rawRDD = train_RDD.map(lambda x: (x[1], x[0], x[2]))

    # userRDD = rawRDD.map(lambda x: x[1]).distinct().zipWithIndex()
    # user_dict = userRDD.collectAsMap()
    # businessRDD = rawRDD.map(lambda x: x[0]).distinct().zipWithIndex()
    # business_dict = businessRDD.collectAsMap()

    # # cleanedRDD: ((user_idx, business_idx), stars)
    # cleanedRDD = rawRDD.map(lambda x: (x[0], (x[1], x[2]))) \
    #     .join(businessRDD).map(lambda x: (x[1][0][0], (x[1][1], x[1][0][1]))) \
    #     .join(userRDD).map(lambda x: ((x[1][1], x[1][0][0]), x[1][0][1])) \
    #     .groupByKey().mapValues(list).mapValues(lambda x: sum(x) / len(x))
    
    # user_business_star = cleanedRDD.collectAsMap()

    # user_business_RDD = cleanedRDD.map(lambda x: x[0]).groupByKey().mapValues(set)
    # user_business_dict = user_business_RDD.collectAsMap()

    # modelRDD = sc.textFile(model_file_path).map(lambda x: json.loads(x))
    # modelRDD = modelRDD.map(lambda x: (x['b1'], (x['b2'], x['sim']))) \
    #     .join(businessRDD).map(lambda x: (x[1][0][0], (x[1][1], x[1][0][1]))) \
    #     .join(businessRDD).map(lambda x: (frozenset((x[1][0][0], x[1][1])), x[1][0][1]))
    
    # model_dict = modelRDD.collectAsMap()

    # def predict(pair_dict):
    #     user_id = pair_dict['user_id']
    #     business_id = pair_dict['business_id']
    #     # cold start for new user or business
    #     if (user_id not in user_dict) or (business_id not in business_dict):
    #         # TODO: cold start problem
    #         # pair_dict['stars'] = avg_dict["UNK"]
    #         pair_dict['stars'] = algo.predict(user_id, business_id)[3]
    #         return pair_dict
    #     user_id_idx = user_dict[user_id]
    #     business_id_idx = business_dict[business_id]

    #     neighbor_set = user_business_dict[user_id_idx]
    #     neighbor_model_list = []
    #     for one in neighbor_set:
    #         pair = frozenset((one, business_id_idx))
    #         if pair in model_dict and abs(model_dict[pair]) >= sim_threshold:
    #             neighbor_model_list.append((one, model_dict[pair]))

    #     # control the number of neighbor
    #     if len(neighbor_model_list) < neighbor_threshold:
    #         # TODO: what if too less neighbor
    #         # if (business_id not in avg_dict):
    #         #     pair_dict['stars'] = avg_backup_dict[business_id]
    #         # else:
    #         #     pair_dict['stars'] = avg_dict[business_id]
    #         pair_dict['stars'] = algo.predict(user_id, business_id)[3]
    #         return pair_dict
        
    #     neighbor_model_list.sort(key = lambda x: abs(x[1]), reverse=True)
    #     # neighbor_model_list = neighbor_model_list[:num_neighbor]

    #     numerator, denominator = 0, 0
    #     for one in neighbor_model_list:
    #         star = user_business_star[(user_id_idx, one[0])]
    #         if one[1] < 0:
    #             star = 6 - star
    #         numerator += star * abs(one[1])
    #         denominator += abs(one[1])
        
    #     pair_dict['stars'] = numerator / denominator
    #     return pair_dict
    
    # # pair = json.loads('{"user_id": "2X7hyChBNMkRfp2X7QEuhA", "business_id": "Gh1BoQNMGkh91pSHqvDRAA"}')
    # # predict(pair)


    # test_predict_RDD = sc.textFile(test_file_path).map(lambda x: json.loads(x)) \
    #     .map(predict).filter(lambda x: "stars" in x)
    
    # predict_result = test_predict_RDD.collect()
    # print("Number of the predicted pairs by item-based CF: {}".format(len(predict_result)))

    # with open(output_file_path, 'w') as f:
    #     for one in predict_result:
    #         print(json.dumps(one), file=f)
    
    # print("Duration: {0:.2f}".format(time.time() - start_time))