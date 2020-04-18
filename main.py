import time
from dataprocessor import DataProcessor
import _k_means_ray
import _k_means_elkan
import _k_means_fast
import _k_means_spark

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import folium
import pytz as tz  # better alternatives -> Apache arrow or pendulum
from datetime import datetime
from PIL import Image
import urllib
import urllib.request
import wget
import ray
from scipy.spatial import Voronoi
from numpy import array

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import joblib
from ray.util.joblib import register_ray

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
import pyspark.mllib.clustering


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

    def dataprocess(self):
        # dataprocessor = DataProcessor(
        #     working_dir, self.input_file)

        df = self.dataprocessor.load_date()
        df = self.dataprocessor.data_filter(df)
        df = self.dataprocessor.data_process(df)
        # df = df.sample(n=2000, replace=False).reset_index(drop=True)
        # config: data 30000 cluster_k: 20
        df = df[:self.sample]
        df_kmeans = df.copy()
        self.df = df_kmeans[['lat', 'lon']]

    def cluster_ray(self, batch_num, init_method="k-means++", assign_method="elkan"):

        # split data
        batches = _k_means_ray.data_split(self.df, num=batch_num)

        # init center
        center = _k_means_ray._k_init(
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

        for i in range(self.iteration):
            # broadcast center point
            for mapper in mappers:
                mapper.broadcastCentroid.remote(center)
                if(assign_method == "elkan" or assign_method == "mega_elkan"):
                    mapper.broadcastDistMatrix.remote(distMatrix)

            # map function
            for mapper in mappers:
                mapper.assign_cluster.remote(method=assign_method)

            newCenter, cost = _k_means_ray.CreateNewCluster(reducers)
            changed, cost_1 = _k_means_ray.ifUpdateCluster(
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
        ray.init(use_pickle=True)
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
        self.dataprocessor.data_transfer(self.df, output_file)
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

    def datapresent(self):
        # print(self.df.shape)
        cluster = self.center
        # cluster[:10]
        #points = np.array([[c[1], c[0]] for c in clusters])
        points = cluster

        # compute Voronoi tesselation
        vor = Voronoi(points)

        # compute regions
        regions, vertices = self.dataprocessor.voronoi_polygons_2d(vor)

        # prepare figure
        plt.style.use('seaborn-white')
        fig = plt.figure()
        fig.set_size_inches(20, 20)

        #geomap
        self.dataprocessor.geomap(self.df, self.df, 13, 2, 'k', 0.1)

        # centroids
        plt.plot(points[:, 0], points[:, 1], 'wo', markersize=10)

        # colorize
        for region in regions:
            polygon = vertices[region]
            plt.fill(*zip(*polygon), alpha=0.4)

        plt.show()


if __name__ == '__main__':
    working_dir = '/Users/wangyifan/Google Drive/checkin'
    input_file = 'loc-gowalla_totalCheckins.txt'
    pipeline = Pipeline(working_dir, input_file, sample=250000,
                        cluster_k=20, iteration=10)
    pipeline.dataprocess()
    # pipeline.cluster_ray(
    #     batch_num=5, init_method="random", assign_method="mega_elkan")
    # pipeline.cluster_sklearn(init_method="k-means++",
    #                          assign_method="full", n_jobs=1)
    pipeline.cluster_spark(output_file='test.txt',
                           init_method="random", epsilon=1e-4)

    # pipeline.datapresent()
    
