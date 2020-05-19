import numpy as np
import sys


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


def isUpdateCluster(newCenter, oldCenter, epsilon=1e-4):
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
