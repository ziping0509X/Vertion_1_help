import numpy as np
import math


def add_users(n):
    position = []
    for i in range(n):
        ind1 = np.random.randint(0, 500)
        ind2 = np.random.randint(0, 500)

        start_position = [ind1, ind2]
        position.append(start_position)
    return position

def get_distanceA(positionA):
    distanceA = np.zeros(len(positionA))
    for i in range(len(positionA)):
        distance = math.sqrt((positionA[i][0] - 250) ** 2 + (positionA[i][1] - 250) ** 2)
        distanceA[i] = distance
    return distanceA

def get_pathlossA(positionA, distanceA):
    PathLoss = np.zeros((len(positionA)))
    for i in range(len(positionA)):
        PathLoss[i] = 128.1 + 37.6 * np.log10(np.sqrt(distanceA[i] ** 2 + (25 - 1.5) ** 2) / 1000)
    return PathLoss

positionA = add_users(5)
positionA = np.array(positionA)
distanceA = get_distanceA(positionA)
pathlossa = get_pathlossA(positionA,distanceA)

print(pathlossa)