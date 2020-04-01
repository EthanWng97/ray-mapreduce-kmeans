import numpy as np
import ray

def data_split(df, seed=None, num=2):
    np.random.seed(43)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    data_1_end = int((1/num) * m)
    data_1 = df.iloc[perm[:data_1_end]]
    data_2 = df.iloc[perm[data_1_end:]]
    return (data_1, data_2)


def randCent(data_X, k):
    n = data_X.shape[1]  # dimension of feature
    centroids = np.empty((k, n))  # matrix of center point
    for j in range(n):
        minJ = min(data_X.iloc[:, j])
        rangeJ = float(max(data_X.iloc[:, j] - minJ))

        centroids[:, j] = (minJ + rangeJ * np.random.rand(k, 1)).flatten()
    return centroids


@ray.remote
def calEDist(arrA, arrB):
    return np.math.sqrt(sum(np.power(arrA-arrB, 2)))

@ray.remote
class KMeansMapper(object):
    centroids = 0

    def __init__(self, item, k=1):
        self.item = item
        self._k = k
        self._clusterAssment = None
        self.centroids = None

    def broadcast(self, centroids):
        self.centroids = centroids

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

            # for each k, calculate the nearest distance
            for j in range(self._k):
                arrA = self.centroids[j, :]
                arrB = self.item[i, :]
                # print(np.math.sqrt(sum(np.power(arrA-arrB, 2))))
                # distJI = self._calEDist(arrA, arrB)
                # distJI = ray.get(calEDist.remote(arrA, arrB))
                distJI = np.math.sqrt(sum(np.power(arrA-arrB, 2)))
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if self._clusterAssment[i, 0] != minIndex or self._clusterAssment[i, 1] > minDist**2:
                self._clusterAssment[i, :] = int(minIndex), minDist**2


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
        return (self._centroids, self._value)

