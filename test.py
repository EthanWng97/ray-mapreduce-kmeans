import time
from dataprocessor import DataProcessor
from raykmeans import *
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
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
import joblib
from ray.util.joblib import register_ray

from scipy.spatial import Voronoi



dataprocessor = DataProcessor(
    '/Users/wangyifan/Google Drive/checkin', 'loc-gowalla_totalCheckins.txt')

df = dataprocessor.load_date()
df = dataprocessor.data_filter(df)
df = dataprocessor.data_process(df)
df_kmeans = df.copy()
df_kmeans = df_kmeans[['lat', 'lon']]
items = data_split(df_kmeans)
center = randCent(df_kmeans, 100)

ray.init()
mappers = [KMeansMapper.remote(item.values, 100) for item in items]

mappers[0].broadcast.remote(center)
mappers[1].broadcast.remote(center)

start = time.time()
mappers[0].assign_cluster.remote()
mappers[1].assign_cluster.remote()
print(ray.get(mappers[0].read_item.remote()))
print(ray.get(mappers[1].read_item.remote()))
end = time.time()
print('execution time: ' + str(end-start) + 's')


# ml = KMeans(n_clusters=100,  init='k-means++', verbose=10, n_jobs=-1)
# # ml.fit(df_kmeans[['lat', 'lon']].sample(frac=0.3))
# ray.init(use_pickle=True)
# register_ray()
# with joblib.parallel_backend('ray'):
#     ml.fit(df_kmeans[['lat', 'lon']].sample(frac=0.3))

# print(df_kmeans.shape)
# cluster = ml.cluster_centers_
# cluster[:10]
# #points = np.array([[c[1], c[0]] for c in clusters])
# points = cluster

# # compute Voronoi tesselation
# vor = Voronoi(points)

# # compute regions
# regions, vertices = voronoi_polygons_2d(vor)

# # prepare figure
# plt.style.use('seaborn-white')
# fig = plt.figure()
# fig.set_size_inches(20, 20)

# #geomap
# dataprocessor.geomap(df_kmeans[['lat', 'lon']], df_kmeans, 13, 2, 'k', 0.1)

# # centroids
# plt.plot(points[:, 0], points[:, 1], 'wo', markersize=10)

# # colorize
# for region in regions:
#     polygon = vertices[region]
#     plt.fill(*zip(*polygon), alpha=0.4)

# plt.show()
