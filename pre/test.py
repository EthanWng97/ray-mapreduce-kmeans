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



# data process
dataprocessor = DataProcessor(
    '/Users/wangyifan/Google Drive/checkin', 'loc-gowalla_totalCheckins.txt')

df = dataprocessor.load_date()
df = dataprocessor.data_filter(df)
df = dataprocessor.data_process(df)
# df = df.sample(n=2000, replace=False).reset_index(drop=True)
# config: data 30000 cluster_k: 20
df = df[:30000]
df_kmeans = df.copy()
df_kmeans = df_kmeans[['lat', 'lon']]
batch_num = 10
cluster_k = 20
epsilon = 1e-4
precision = 1e-6
iteration = 10
        
"""
RAY + MAPREDUCE METHOD
"""
# split data
# items = _k_means_ray.data_split(df_kmeans, num=batch_num)

# # init center
# center = _k_means_ray._k_init(df_kmeans, cluster_k, method="k-means++")
# print(center)
# n = center.shape[0]  # n center points
# distMatrix = np.empty(shape=(n, n))
# _k_means_fast.createDistMatrix(center, distMatrix)

# # init ray
# ray.init()
# mappers = [_k_means_ray.KMeansMapper.remote(
#     item.values, k=cluster_k) for item in items[0]]
# reducers = [_k_means_ray.KMeansReducer.remote(
#     i, *mappers) for i in range(cluster_k)]
# start = time.time()

# for i in range(iteration):
#     # broadcast center point
#     for mapper in mappers:
#         mapper.broadcastCentroid.remote(center)
#         mapper.broadcastDistMatrix.remote(distMatrix)
#     # print(distMatrix)
#     # map function
#     for mapper in mappers:
#         mapper.assign_cluster.remote()

#     # print(ray.get(mappers[0].read_cluster.remote()))
#     # for mapper in mappers:
#     #     print(ray.get(mapper.read_cluster.remote()))
#     # reduce function
#     # for reducer in reducers:
#     #     print(ray.get(reducer.update_cluster.remote()))
#     newCenter = _k_means_ray.CreateNewCluster(reducers)
#     # print(newCenter)
#     # newCenter = ray.get(reducer.update_cluster.remote())
#     # print(newCenter)
#     # print(center)
#     changed, cost = _k_means_ray.ifUpdateCluster(newCenter, center) # update
#     if (not changed):
#         break
#     else:
#         center = newCenter
#         _k_means_fast.createDistMatrix(center, distMatrix)
#         # distMatrix = _k_means_fast.createDistMatrix(center)
#         print(str(i) + " iteration, cost: "+ str(cost))

# # print(center)
# end = time.time()
# print(center)
# print('execution time: ' + str(end-start) + 's, cost: '+ str(cost))



"""
SKLEARN METHOD
"""
# from sklearn.cluster import KMeans
# from sklearn import metrics
# from sklearn.metrics import pairwise_distances
# import joblib
# from ray.util.joblib import register_ray
# start = time.time()

# ml = KMeans(n_clusters=cluster_k,  init='random', verbose=1,
#             n_jobs=1, max_iter=10, algorithm='full')
# ml.fit(df_kmeans)

# end = time.time()
# print('execution time: ' + str(end-start) + 's')
# ray.init(use_pickle=True)
# register_ray()
# with joblib.parallel_backend('ray'):
#     ml.fit(df_kmeans[['lat', 'lon']].sample(frac=0.3))



"""
SPARK METHOD
"""

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.mllib.clustering import KMeans, KMeansModel
dataprocessor.data_transfer(df)

# def loadDataset(df):
#     # df = pd.read_csv(infile, sep='\t', header=0, dtype=str, na_filter=False)
#     return np.array(df).astype(np.float)

sc = SparkContext(appName="KmeansExample")
# spark = SparkSession.builder.appName('Recommendation_system').getOrCreate()
# # data_X = loadDataset(df_kmeans[['lat', 'lon']])
# data_X = spark.createDataFrame(df_kmeans[['lat', 'lon']])

# clusters = KMeans.train(
#     data_X.rdd, k=20, maxIterations=10, initializationMode="random")
# print(clusters.clusterCenters)
# sc = SparkContext(appName="KmeansExample")
    
data = sc.textFile(
        "/Users/wangyifan/Desktop/CP3106-Independent-Project-NUS/test.txt")
parsedData = data.map(lambda line: array(
       [float(x) for x in line.split('\t')]))

# Build the model (cluster the data)
clusters = KMeans.train(parsedData, 20, maxIterations=10,
                           initializationMode="random")
print(clusters.centers)
center = np.array(clusters.centers)
print(center)
print(type(center))


















# print(df_kmeans.shape)
cluster = center
cluster[:10]
#points = np.array([[c[1], c[0]] for c in clusters])
points = cluster

# compute Voronoi tesselation
vor = Voronoi(points)

# compute regions
regions, vertices = dataprocessor.voronoi_polygons_2d(vor)

# prepare figure
plt.style.use('seaborn-white')
fig = plt.figure()
fig.set_size_inches(20, 20)

#geomap
dataprocessor.geomap(df_kmeans[['lat', 'lon']], df_kmeans, 13, 2, 'k', 0.1)

# centroids
plt.plot(points[:, 0], points[:, 1], 'wo', markersize=10)

# colorize
for region in regions:
    polygon = vertices[region]
    plt.fill(*zip(*polygon), alpha=0.4)

plt.show()
