import tensorflow as tf
import numpy as np
import random
import math

NUMA = 5
NUMB = 10
LENTH = 500
HEITH = 500
positionBS = [250,250]

class ENVIRONMENT:
    def __init__(self,NUMA,NUMB):
        self.numA = NUMA
        self.numB = NUMB
        self.positionA = self.add_users(self.numA)
        self.positionB = self.add_users(self.numB)
        self.distanceA = self.get_distanceA(self.positionA)
        self.distanceB = self.get_distanceB(self.positionB)
        self.pathlossA = self.get_pathlossA(self.positionA, self.distanceA)
        self.pathlossA = self.get_pathlossA(self.positionB, self.distanceB)
        #得到了A\B类用户的坐标、相互之间的距离、路径损耗数值

    def add_users(self,n):
        position = []
        for i in range(n):
            ind1 = np.random.randint(0, 500)
            ind2 = np.random.randint(0, 500)

            start_position = [ind1, ind2]
            position.append(start_position)
        return position

    def get_distanceA(self,positionA):
        distanceA = []
        for i in range(len(positionA)):
            distance = math.sqrt((positionA[i][0] - 250) ** 2 + (positionA[i][1] - 250) ** 2)
            distanceA.append(distance)
        return distanceA

    def get_distanceB(self,positionB):
        distanceB = np.zeros([len(positionB), len(positionB)])
        for i in range(len(positionB)):
            for j in range(len(positionB)):
                if not i == j:
                    distanceB[i, j] = math.sqrt(
                        (positionB[i][0] - positionB[j][0]) ** 2 + (positionB[i][1] - positionB[j][1]) ** 2)
        return distanceB

    def get_pathlossA(self,positionA, distanceA):
        PathLoss = np.zeros((len(positionA)))
        for i in range(len(positionA)):
            PathLoss[i] = 128.1 + 37.6 * np.log10(math.sqrt(distanceA[i] ** 2 + (25 - 1.5) ** 2) / 1000)
        return PathLoss

    def get_pathlossB(self,positionB, distanceB):
        PathLoss = np.zeros(shape=(len(positionB), len(positionB)))

        for i in range(len(positionB)):
            for j in range(len(positionB)):
                PathLoss[i][j] = 128.1 + 37.6 * np.log10(math.sqrt(distanceB[i][j] ** 2 + (25 - 1.5) ** 2) / 1000)

        return PathLoss


