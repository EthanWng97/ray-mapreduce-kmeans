import numpy as np
import ray
import sys
import _k_means_elkan
import _k_means_fast
import _k_means_spark


def _k_init(data_X, n_clusters, method="k-means++"):
    n = data_X.shape[1]  # dimension of feature
    centroids = np.empty((n_clusters, n))  # matrix of center point
    if(method=="k-means++"):
        print("trying k-means++ method to initialize k clusters")
        data = data_X.copy()
        total = data.shape[0] # # samples
        index_n = np.arange(total)
        prob_n = np.empty(shape=(1, total),dtype=np.float32)

        center_1 = np.random.randint(0, data.shape[0])
        centroids[0] = data.loc[center_1]

        for i in range(1, n_clusters):
            index_row = 0
            index = 0
            # index_max = 0
            # maxDist = 0
            totalDist = 0
            # calaulate proper point
            for row in data.values:
                # calculate shortest distance (D(x)) for each point
                minDistJ = np.inf
                for j in range(i):
                    distJ = calEDist(row, centroids[j])

                    if distJ < minDistJ:
                        minDistJ = distJ
                totalDist += minDistJ
                prob_n[0][index_row] = minDistJ
                # if minDistJ > maxDist:
                #     maxDist = minDistJ
                #     index_max = index_row

                #operate index and maxDist
                index_row += 1
            prob_n = prob_n/totalDist
            index = np.random.choice(index_n, p=prob_n[0].ravel())
            centroids[i] = data.loc[index]
    
    elif(method=="random"):
        print("trying random method to initialize k clusters")

        for k in range(n):
            minK = min(data_X.iloc[:, k])
            rangeK = float(max(data_X.iloc[:, k] - minK))

            centroids[:, k] = (
                minK + rangeK * np.random.rand(n_clusters, 1)).flatten()
    else:
        print("run failed: wrong method of initializing k clusters")
        sys.exit(2)
    return centroids



def data_split(df, seed=None, num=3):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    data_end = np.zeros(shape=(1, num-1), dtype=np.int)
    for i in range(num-1):
        data_end[0][i] = int(((i+1)/num)*m)
    data = np.zeros(shape=(1, num), dtype=object)
    for i in range(num):
        if (i == 0):
            data[0][i] = df.iloc[perm[:data_end[0][0]]]
        elif (i == num-1):
            data[0][i] = df.iloc[perm[data_end[0][i-1]:]]
        else:
            data[0][i] = df.iloc[perm[data_end[0][i-1]:data_end[0][i]]]
    return tuple(data)


def data_split_seq(df, num=3):
    m = len(df.index)
    data_end = np.zeros(shape=(1, num-1), dtype=np.int)
    for i in range(num-1):
        data_end[0][i] = int(((i+1)/num)*m)
    data = np.zeros(shape=(1, num), dtype=object)
    for i in range(num):
        if (i == 0):
            data[0][i] = df.iloc[:data_end[0][0]]
        elif (i == num-1):
            data[0][i] = df.iloc[data_end[0][i-1]:]
        else:
            data[0][i] = df.iloc[data_end[0][i-1]:data_end[0][i]]
    return tuple(data)

# def randCent(data_X, n_clusters):
#     n = data_X.shape[1]  # dimension of feature
#     centroids = np.empty((n_clusters, n))  # matrix of center point
#     for j in range(n):
#         minJ = min(data_X.iloc[:, j])
#         rangeJ = float(max(data_X.iloc[:, j] - minJ))

#         centroids[:, j] = (
#             minJ + rangeJ * np.random.rand(n_clusters, 1)).flatten()
#     return centroids

def calEDist(arrA, arrB):
    return np.math.sqrt(sum(np.power(arrA-arrB, 2)))

def CalculateNorm(point):
    return np.linalg.norm(point)

def ifUpdateCluster(newCenter, oldCenter, epsilon=1e-4):
    changed = False
    if (newCenter.shape[0] != oldCenter.shape[0]):
        print("run failed: no matched dimension about newCenter and oldCenter list!")
        sys.exit(2)
    n = newCenter.shape[0]
    cost = 0
    for i in range(n):
        diff = _k_means_spark.fastSquaredDistance(newCenter[i], CalculateNorm(
            newCenter[i]), oldCenter[i], CalculateNorm(oldCenter[i]))
        if diff > np.square(epsilon):
            changed = True
        cost += diff
    return changed, cost

def CreateNewCluster(reducers):
    cost = 0
    new_cluster = np.array([[0., 0.]])
    for reducer in reducers:
        tmp = ray.get(reducer.update_cluster.remote())
        new_cluster = np.insert(
            new_cluster, 0, tmp, axis=0)
        cost += ray.get(reducer.read_cost.remote())
    return np.delete(new_cluster, -1, axis=0), cost



@ray.remote
class KMeansMapper(object):
    centroids = 0

    def __init__(self, item, k=1, epsilon=1e-4, precision=1e-6):
        self.item = item
        self._k = k
        self._clusterAssment = None
        self.centroids = None
        self._epsilon = epsilon
        self._precision = precision
        self._distMatrix = None

    def broadcastCentroid(self, centroids):
        self.centroids = centroids

    def broadcastDistMatrix(self, distMatrix):
        self._distMatrix = distMatrix

    def _calEDist(self, arrA, arrB):
        return np.math.sqrt(sum(np.power(arrA-arrB, 2)))

    def read_cluster(self):
        return self._clusterAssment

    def read_item(self):
        return self.item

    def assign_cluster(self, method="elkan"):
        m = self.item.shape[0]  # number of sample
        self._clusterAssment = np.zeros((m, 2))
        # assign nearest center point to the sample
        for i in range(m):
            minDist = np.inf
            minIndex = -1

            if(method == "spark"):
                """
                method 1: optimize findclosest center
                """
                minIndex, minDist = _k_means_spark.findClosest(
                    self._k, self.centroids, self.item, i, self._epsilon, self._precision)

            elif(method == "full"):
                """
                method 2: classic calculation method
                """
                # for each k, calculate the nearest distance
                for j in range(self._k):
                    arrA = self.centroids[j, :]
                    arrB = self.item[i, :]
                    distJI = calEDist(arrA, arrB)
                    # distJI = np.math.sqrt(sum(np.power(arrA-arrB, 2)))
                    if distJI < minDist:
                        minDist = distJI
                        minIndex = j
            
            elif(method == "elkan"):
                """
                method 3: elkan method
                """
                minIndex, minDist = _k_means_elkan.findClosest(
                    self._k, self.centroids, self.item, i, self._distMatrix)
            else:
                print("run failed: wrong algorithm for assigning point")
                sys.exit(2)

            # output: minIndex, minDist
            # if self._clusterAssment[i, 0] != minIndex or self._clusterAssment[i, 1] > minDist:
            self._clusterAssment[i, :] = int(minIndex), minDist


@ray.remote
class KMeansReducer(object):
    def __init__(self, value, *kmeansmappers):
        self._value = value
        self.kmeansmappers = kmeansmappers
        self.centroids = None # recalculated center point
        self._clusterAssment = None
        self._clusterOutput = np.array([[0., 0.]])
        self._cost = 0

    def read(self):
        return self._value
    
    def read_cost(self):
        return self._cost
    
    def update_cluster(self):
        self._cost = 0
        for mapper in self.kmeansmappers:
            self._clusterAssment = ray.get(mapper.read_cluster.remote())
            # get index number of each sample
            index_all = self._clusterAssment[:, 0]  

            self._cost += np.sum(self._clusterAssment[:, 1])
            # filter the sample according to the reducer number
            value = np.nonzero(index_all == self._value)

            # ray.get(mapper.read_item.remote)
            # get the info of sample according to the reducer number
            ptsInClust = ray.get(mapper.read_item.remote())[
                value[0]]
            
            # accumulate the result
            # self._clusterOutput = np.append(self._clusterOutput, ptsInClust)
            self._clusterOutput = np.insert(
                self._clusterOutput, 0, ptsInClust, axis=0)
        
        try:
            self._clusterOutput = np.delete(self._clusterOutput, -1, axis=0)
        except IndexError:
            print("run failed: incorrect mapper data!")
            sys.exit(2)
        else:
            # calculate the mean of all samples
            self._centroids = np.mean(self._clusterOutput, axis=0)
            # return (self._centroids, self._value)
            return self._centroids
