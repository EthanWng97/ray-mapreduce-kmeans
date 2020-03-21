from __future__ import print_function
import pandas as pd
from numpy import array
from math import sqrt
import numpy as np
from pyspark import SparkContext

from pyspark.mllib.clustering import KMeans, KMeansModel


def loadDataset(infile):
    df = pd.read_csv(infile, sep='\t', header=0, dtype=str, na_filter=False)
    return np.array(df).astype(np.float)

if __name__ == "__main__":
    sc = SparkContext(appName="KmeansExample")

    # Load and parse the data
    data = sc.textFile("/Users/wangyifan/Google Drive/testSet.txt")
    parsedData = data.map(lambda line: array(
        [float(x) for x in line.split(' ')]))
    data_X = loadDataset(r"/Users/wangyifan/Google Drive/testSet.txt")

    # Build the Model(cluster the data)
    clusters = KMeans.train(
        data_X, 2, maxIterations=10, initializationMode="random")
    print(clusters.clusterCenters)

    print(clusters.predict([0.2, 0.2, 0.2]))

    # Evaluate clustering by computing Within Set Sum of Squared Errors
    def error(point):
        center = clusters.centers[clusters.predict(point)]
        return sqrt(sum([x**2 for x in (point - center)]))

    WSSSE = data_X.map(lambda point: error(point)
                           ).reduce(lambda x, y: x + y)
    print("Within Set Sum of Squared Error = " + str(WSSSE))
