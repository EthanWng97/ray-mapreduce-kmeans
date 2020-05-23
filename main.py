from utils import _k_means_ray
from utils import _k_means_elkan
from utils import _k_means_fast
from utils import _k_means_spark
from utils.dataprocessor import DataProcessor

import os
import sys
import time
import getopt
import numpy as np
import ray
from numpy import array

import joblib
from sklearn.cluster import KMeans
from ray.util.joblib import register_ray

from pyspark import SparkContext
import pyspark.mllib.clustering

# d:f:s:k:n:m:t:
def usage():
    print("usage: " +
          sys.argv[0] + " -d working-dir -f input-file -s number-of-sample -k number-of-clusters -n number-of-iteration -m number-of-mappers -t number-of-tasks")


class Pipeline:
    def __init__(self, working_dir, input_file, sample=None, cluster_k=20, iteration=10):
        self.input_file = input_file
        self.sample = sample
        self.cluster_k = cluster_k
        self.iteration = iteration
        self.df = None
        self.center = None
        self.dataprocessor = DataProcessor(
            working_dir, input_file)
        self.df = self.dataprocessor.processData(sample)

    def cluster_ray(self, batch_num, init_method="k-means++", assign_method="elkan" ,task_num=2):

        # split data
        batches = _k_means_ray.splitData(self.df, num=batch_num)

        # init center
        center = _k_means_ray._initK(
            self.df, self.cluster_k, method=init_method)
        print(center)
        n = center.shape[0]  # n center points
        distMatrix = np.empty(shape=(n, n))
        _k_means_fast.createDistMatrix(center, distMatrix)

        # init ray
        ray.init()
        mappers = [_k_means_ray.KMeansMapper.remote(
            mini_batch.values, k=self.cluster_k) for mini_batch in batches[0]]
        reducers = [_k_means_ray.KMeansReducer.remote(
            i, *mappers) for i in range(self.cluster_k)]
        start = time.time()
        cost = 0

        for i in range(self.iteration):
            # broadcast center point
            for mapper in mappers:
                mapper.broadcastCentroid.remote(center)
                if(assign_method == "elkan" or assign_method == "mega_elkan"):
                    mapper.broadcastDistMatrix.remote(distMatrix)

            # map function
            for mapper in mappers:
                mapper.assignCluster.remote(
                    method=assign_method, task_num=task_num)

            newCenter, cost = _k_means_ray.createNewCluster(reducers)
            changed, cost_1 = _k_means_ray.isUpdateCluster(
                newCenter, center)  # update
            if (not changed):
                break
            else:
                center = newCenter
                if(assign_method == "elkan" or assign_method == "mega_elkan"):
                    _k_means_fast.createDistMatrix(center, distMatrix)
                print(str(i) + " iteration, cost: " + str(cost))

        # print(center)
        end = time.time()
        print(center)
        self.center = center
        print('execution time: ' + str(end-start) + 's, cost: ' + str(cost))

    def cluster_sklearn(self, init_method="k-means++", assign_method="elkan", n_jobs=1):
        start = time.time()

        ml = KMeans(n_clusters=self.cluster_k,  init=init_method, verbose=1,
                    n_jobs=n_jobs, max_iter=self.iteration, algorithm=assign_method)

        ml.fit(self.df)
        # ray.init(use_pickle=True)
        # register_ray()
        # with joblib.parallel_backend('ray'):
        #     ml.fit(self.df.sample(n=self.sample))
        end = time.time()
        center = ml.cluster_centers_
        print(center)
        self.center = center

        print('execution time: ' + str(end-start) + 's')

    def cluster_spark(self, output_file='test.txt', init_method="random", epsilon=1e-4):
        start = time.time()
        output_name = './data/' + output_file
        self.dataprocessor.saveData(self.df, output_file)
        sc = SparkContext(appName="KmeansSpark")
        data = sc.textFile(output_name)
        parsedData = data.map(lambda line: array(
            [float(x) for x in line.split('\t')]))

        # Build the model (cluster the data)
        clusters = pyspark.mllib.clustering.KMeans.train(parsedData, k=self.cluster_k, maxIterations=self.iteration,
                                                         initializationMode=init_method, epsilon=epsilon)
        end = time.time()
        center = np.array(clusters.centers)
        print(center)
        self.center = center
        print('execution time: ' + str(end-start) + 's')

# if __name__ == '__main__':
#     working_dir = '/Users/wangyifan/Google Drive/checkin'
#     input_file = 'loc-gowalla_totalCheckins.txt'
#     pipeline = Pipeline(working_dir, input_file, sample=50000,
#                         cluster_k=20, iteration=0)
#     pipeline.cluster_ray(
#         batch_num=5, init_method="random", assign_method="mega_elkan", task_num=2)
    # pipeline.cluster_sklearn(init_method="k-means++",
    #                          assign_method="full", n_jobs=1)
    # pipeline.cluster_spark(output_file='test.txt',
    #                        init_method="random", epsilon=1e-4)

    # pipeline.dataprocessor.presentData(pipeline.center, pipeline.df)

working_dir = input_file = None
number_of_sample = 500000
number_of_clusters = 20
number_of_iteration = 10
number_of_mappers = 5
number_of_tasks = 2

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:f:s:k:n:m:t:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        working_dir = a
    elif o == '-f':
        input_file = a
    elif o == '-s':
        number_of_sample = int(a)
    elif o == '-k':
        number_of_clusters = int(a)
    elif o == '-n':
        number_of_iteration = int(a)
    elif o == '-m':
        number_of_mappers = int(a)
    elif o == '-t':
        number_of_tasks = int(a)
    else:
        assert False, "unhandled option"

if working_dir == None or input_file == None:
    usage()
    sys.exit(2)

pipeline = Pipeline(working_dir, input_file, sample=number_of_sample,
                    cluster_k=number_of_clusters, iteration=number_of_iteration)
pipeline.cluster_ray(
    batch_num=number_of_mappers, init_method="random", assign_method="mega_elkan", task_num=number_of_tasks)
