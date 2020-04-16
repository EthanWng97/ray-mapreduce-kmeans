import numpy as np
import ray

def createDistMatrix(center):
    n = center.shape[0]  # n center points
    rst = np.empty(shape=(n, n), dtype=np.float32)
    for i in range(n-1):
        rst[i][i+1] = np.math.sqrt(sum(np.power(center[i]-center[i+1], 2)))

    return rst


def calEDist(point, centroids, distMatrix, k):
    bestDistance = np.inf
    bestIndex = -1
    j = 0
    # print(distMatrix[12][13])
    # for each k(k centers)
    while(j < k):
        center = centroids[j, :]
        distJI = np.math.sqrt(sum(np.power(center-point, 2)))
        if (distJI < bestDistance):
            bestDistance = distJI
            bestIndex = j

        if (j <= k-2 and 2*distJI <= distMatrix[j][j+1]):  # optimize
            j += 1

        j += 1
    # return np.math.sqrt(sum(np.power(arrA-arrB, 2)))
    return bestIndex, bestDistance


def findClosest(k, centroids, item, i, distMatrix):
    bestDistance = np.inf
    bestIndex = -1
    j = 0
    # print(distMatrix[12][13])
    # for each k(k centers)
    while(j < k):
        center = centroids[j, :]
        point = item[i, :]
        distJI = np.math.sqrt(sum(np.power(center-point, 2)))
        # distJI = calEDist(center, point)
        if (distJI < bestDistance):
            bestDistance = distJI
            bestIndex = j
        
        if (j<=k-2 and 2*distJI <= distMatrix[j][j+1]):  # optimize
            j += 1

        j += 1
        # print(j)

    
    return bestIndex, bestDistance

@ray.remote
def mega_findClosest(k, centroids, distMatrix, item):
    n = item.shape[0] # n samples after splilt
    rst = np.empty(shape=(n, 2),dtype=np.float32)
    for i in range(n):
        bestIndex, bestDistance = calEDist(
            item[i, :], centroids, distMatrix, k)
        rst[i, :] = bestIndex, bestDistance

    return rst

