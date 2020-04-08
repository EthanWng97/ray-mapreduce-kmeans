import numpy as np

def createDistMatrix(center):
    n = center.shape[0]  # n center points
    rst = np.empty(shape=(n, n), dtype=np.float32)
    for i in range(n-1):
        rst[i][i+1] = np.math.sqrt(sum(np.power(center[i]-center[i+1], 2)))

    #     j = i
    #     while(j < n-1):
    #         j += 1
    #         rst[i][j] = np.math.sqrt(sum(np.power(center[i]-center[j], 2)))

    # for i in range(n):
    #     j = i
    #     # print(i)
    #     while(j > 0):
    #       j -= 1
    #       # print(j)
    #       rst[i][j] = rst[j][i]

    return rst


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
        if (distJI < bestDistance):
            bestDistance = distJI
            bestIndex = j
        # print(distMatrix[j][j+1])
        if (j<=k-2 and 2*distJI <= distMatrix[j][j+1]):  # optimize
            j += 1

        j += 1
        # print(j)

    
    return bestIndex, bestDistance

