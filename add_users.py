import numpy as np
import tensorflow as tf
import random
import math

#============================
# 二维平面是500*500
# 生成A、B类用户的坐标、相互之间的距离
# 根据以上数据，生成path_loss\shadow\fastfading的数据(考虑只用path_loss)
# 是坐标变动以后，仍然有学习效果？还是只能在指定坐标下面一步步优化到max
#============================

def add_users(n):
    position = []
    for i in range(n):
        ind1 = np.random.randint(0,500)
        ind2 = np.random.randint(0, 500)

        start_position = [ind1,ind2]
        position.append(start_position)
    return position

def get_distanceA(positionA):
    distanceA = []
    for i in range(len(positionA)):
        distance = math.sqrt((positionA[i][0] - 250)**2 + (positionA[i][1] - 250)**2)
        distanceA.append(distance)
    return distanceA

def get_distanceB(positionB):
    distanceB = np.zeros([len(positionB),len(positionB)])
    for i in range(len(positionB)):
        for j in range(len(positionB)):
            if not i == j:
                distanceB[i,j] = math.sqrt((positionB[i][0] - positionB[j][0])**2 + (positionB[i][1] - positionB[j][1])**2)
    return distanceB

def get_pathlossA(positionA,distanceA):
    PathLoss = np.zeros((len(positionA)))
    for i in range(len(positionA)):
        PathLoss[i] = 128.1 + 37.6*np.log10(math.sqrt(distanceA[i]**2 + (25 - 1.5)**2)/1000)
    return PathLoss

def get_pathlossB(positionB,distanceB):
    PathLoss = np.zeros(shape= (len(positionB),len(positionB)))

    for i in range(len(positionB)):
        for j in range(len(positionB)):
            PathLoss[i][j] = 128.1 + 37.6*np.log10(math.sqrt(distanceB[i][j]**2 + (25 - 1.5)**2)/1000)

    return PathLoss

#def get_shadowA(positionA,distanceA):

N = 5
M = 10
positionBS = [250,250]
positionA = add_users(n= N)
positionB = add_users(n= M)
# print(positionA)
# print(positionB)
distanceA = get_distanceA(positionA)
distanceB = get_distanceB(positionB)
# print(distanceA)
# print(distanceB)
pathlossA = get_pathlossA(positionA,distanceA)
pathlossB = get_pathlossB(positionB,distanceB)

print(pathlossA)
print(pathlossB)


