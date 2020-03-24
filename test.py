from dataprocessor import DataProcessor
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



def voronoi_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Input_args:
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    :returns:
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all([v >= 0 for v in vertices]):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


dataprocessor = DataProcessor(
    '/Users/wangyifan/Google Drive/checkin', 'loc-gowalla_totalCheckins.txt')

df = dataprocessor.load_date()
df = dataprocessor.data_filter(df)
df = dataprocessor.data_process(df)
df_kmeans = df.copy()
ml = KMeans(n_clusters=100, init='k-means++', verbose=10, n_jobs=-1)
# ml.fit(df_kmeans[['lat', 'lon']].sample(frac=0.3))
ray.init(use_pickle=True)
register_ray()
with joblib.parallel_backend('ray'):
    ml.fit(df_kmeans[['lat', 'lon']].sample(frac=0.3))

print(df_kmeans.shape)
cluster = ml.cluster_centers_
cluster[:10]
#points = np.array([[c[1], c[0]] for c in clusters])
points = cluster

# compute Voronoi tesselation
vor = Voronoi(points)

# compute regions
regions, vertices = voronoi_polygons_2d(vor)

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
