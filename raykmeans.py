import numpy as np
import ray
import sys
import _k_means_elkan
import _k_means_fast

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


def randCent(data_X, k):
    n = data_X.shape[1]  # dimension of feature
    centroids = np.empty((k, n))  # matrix of center point
    for j in range(n):
        minJ = min(data_X.iloc[:, j])
        rangeJ = float(max(data_X.iloc[:, j] - minJ))

        centroids[:, j] = (minJ + rangeJ * np.random.rand(k, 1)).flatten()
    return centroids


def calEDist(arrA, arrB):
    return np.math.sqrt(sum(np.power(arrA-arrB, 2)))

def CalculateNorm(point):
    return np.linalg.norm(point)


def fastSquaredDistance(center, center_norm, point, point_norm, EPSILON=1e-4, precision=1e-6):
    n = center.size
    sumSquaredNorm = np.square(center_norm) + np.square(point_norm)
    normDiff = center_norm - point_norm
    sqDist = 0.0
    precisionBound1 = 2.0 * EPSILON * \
        sumSquaredNorm / (np.square(normDiff) + EPSILON)
    if (precisionBound1 < precision):
        sqDist = sumSquaredNorm - 2.0 * np.dot(center, point)
    else:
        sqDist = calEDist(center, point)
    return sqDist


def findClosest(k, centroids, item, i, EPSILON, precision):
    bestDistance = np.inf
    bestIndex = -1
    j = 0
    # for each k(k centers)
    for j in range(k):
        center = centroids[j, :]
        point = item[i, :]
        center_norm = np.linalg.norm(center)
        point_norm = np.linalg.norm(point)
        lowerBoundOfSqDist = center_norm - point_norm
        lowerBoundOfSqDist = np.square(lowerBoundOfSqDist)
        if (lowerBoundOfSqDist < bestDistance):
            distance = fastSquaredDistance(
                center, center_norm, point, point_norm, EPSILON, precision)  # 计算欧氏距离
            if (distance < bestDistance):  # 如果距离小于最优值，那么更新最优值
                bestDistance = distance
                bestIndex = j
    return bestIndex, bestDistance


def ifUpdateCluster(newCenter, oldCenter, epsilon=1e-4):
    changed = False
    if (newCenter.shape[0] != oldCenter.shape[0]):
        print("run failed: no matched dimension about newCenter and oldCenter list!")
        sys.exit(2)
    n = newCenter.shape[0]
    cost = 0
    for i in range(n):
        diff = fastSquaredDistance(newCenter[i], CalculateNorm(
            newCenter[i]), oldCenter[i], CalculateNorm(oldCenter[i]))
        if diff > np.square(epsilon):
            changed = True
        cost += diff
    return changed, cost

def CreateNewCluster(reducers):
    new_cluster = np.array([[0., 0.]])
    for reducer in reducers:
        tmp = ray.get(reducer.update_cluster.remote())
        new_cluster = np.insert(
            new_cluster, 0, tmp, axis=0)
    return np.delete(new_cluster, -1, axis=0)



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

    def assign_cluster(self):
        m = self.item.shape[0]  # number of sample
        self._clusterAssment = np.zeros((m, 2))
        # assign nearest center point to the sample
        for i in range(m):
            minDist = np.inf
            minIndex = -1

            """
            method 1: optimize findclosest center
            """
            # minIndex, minDist = findClosest(
            #     self._k, self.centroids, self.item, i, self._epsilon, self._precision)

            """
            method 2: classicial calculation method
            """

            # for each k, calculate the nearest distance
            # for j in range(self._k):
            #     arrA = self.centroids[j, :]
            #     arrB = self.item[i, :]
            #     distJI = calEDist(arrA, arrB)
            #     # distJI = np.math.sqrt(sum(np.power(arrA-arrB, 2)))
            #     if distJI < minDist:
            #         minDist = distJI
            #         minIndex = j
                    
            """
            method 3: elkan method
            """
            
            minIndex, minDist = _k_means_elkan.findClosest(
                self._k, self.centroids, self.item, i, self._distMatrix)


            # print(minIndex, minDist)
            # output: minIndex, minDist
            if self._clusterAssment[i, 0] != minIndex or self._clusterAssment[i, 1] > minDist**2:
                self._clusterAssment[i, :] = int(minIndex), minDist


@ray.remote
class KMeansReducer(object):
    def __init__(self, value, *kmeansmappers):
        self._value = value
        self.kmeansmappers = kmeansmappers
        self.centroids = None # recalculated center point
        self._clusterAssment = None
        self._clusterOutput = np.array([[0., 0.]])

    def read(self):
        return self._value
    
    def update_cluster(self):
        for mapper in self.kmeansmappers:
            self._clusterAssment = ray.get(mapper.read_cluster.remote())
            # get index number of each sample
            index_all = self._clusterAssment[:, 0]  
            
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
        
        self._clusterOutput = np.delete(self._clusterOutput, -1, axis=0)
        # calculate the mean of all samples
        self._centroids = np.mean(self._clusterOutput, axis=0)
        # return (self._centroids, self._value)
        return self._centroids
