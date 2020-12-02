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
        self.distance_A_B = self.get_distance_A_B(self.positionA,self.positionB)
        self.distance_B_B = self.get_distance_A_B(self.positionB, self.positionB)

        self.pathlossA = self.get_pathlossA(self.positionA, self.distanceA)
        self.pathlossB = self.get_pathlossA(self.positionB, self.distanceB)
        self.pathlossA_B = self.get_pathlossAB(self.distance_A_B)
        self.pathlossB_B = self.get_pathlossBB(self.distance_A_B)

        self.A_power_dB = 23
        self.B_power_list = [5,10,23]
        self.A_Ant_G = 8
        self.A_Noise_g = 5
        self.B_Ant_G = 3
        self.B_Noise_g = 9

        self.sig2_dB = -114
        self.sig2 = 10 ** (self.sig2_dB / 10)
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

    def get_distance_A_B(self,positionA,positionB):
        distance_A_B = np.zeros((len(positionB),len(positionA)))
        for i in range(len(positionB)):
            for j in range(len(positionA)):
                distance_A_B[i,j] = math.sqrt(
                    (positionA[i][0] - positionB[j][0]) ** 2 - (positionA[i][1] - positionB[j][1]) ** 2
                )
        return distance_A_B

    def get_pathlossA(self,positionA, distanceA):
        PathLoss = np.zeros((len(positionA)))
        for i in range(len(positionA)):
            PathLoss[i] = 128.1 + 37.6 * np.log10(math.sqrt(distanceA[i] ** 2 + (25 - 1.5) ** 2) / 1000)
        return PathLoss

    def get_pathlossB(self,positionB, distanceB): #先不用这个互相的pathloss，因为假设B类用户也是向基站通信的
        PathLoss = np.zeros(shape=(len(positionB), len(positionB)))

        for i in range(len(positionB)):
            for j in range(len(positionB)):
                PathLoss[i][j] = 128.1 + 37.6 * np.log10(math.sqrt(distanceB[i][j] ** 2 + (25 - 1.5) ** 2) / 1000)

        return PathLoss

    def get_pathlossAB(self,distance_A_B):
        PathLoss = np.zeros((self.numB,self.numA))

        for i in range (len(self.numB)):
            for j in range(len(self.numA)):
                PathLoss[i,j] = 128.1 + 37.6 * np.log10(math.sqrt(distance_A_B[i][j] ** 2 + (25 - 1.5) ** 2) / 1000)

        return PathLoss

    def get_pathlossBB(self, distance_A_B):
        PathLoss = np.zeros((self.numB, self.numB))

        for i in range(len(self.numB)):
            for j in range(len(self.numB)):
                PathLoss[i, j] = 128.1 + 37.6 * np.log10(math.sqrt(distance_A_B[i][j] ** 2 + (25 - 1.5) ** 2) / 1000)

        return PathLoss

    def get_state(self):
        #self.state_input = []
        self.pathlossA = np.array(self.pathlossA)
        self.pathlossB = np.array(self.pathlossB)
        self.state_input = np.append(self.pathlossA, self.pathlossB)

        return self.state_input

    def get_reward(self,stateinput,actionall,idx): #单词循环中给单用户选择动作得到的奖励，如何设置？
        #action输入的是一个数字，5*3 = 15
        actionall_1 = actionall.copy()
        RB_select = actionall[:,0] #[1选择的RB，2选择的RB，......,10选择的RB]
        Power_select = actionall[:,1] #[1选择的P，2选择的P，......,10选择的P]
        B_signal = np.zeros(self.numB)
        BB_interference = np.zeros((self.numB))
        #去掉自己，方便后面大循环
        RB_select[idx] = 100
        #下面开始计算B_signal、BB_interference

        for i in range(self.numB)
            indexes = np.argwhere(RB_select == i) #返回了一个一维数组

            for j in range(len(indexes)):
                B_signal[j] = \
                    10**(0.1 * (self.B_power_list[Power_select[j]] -
                                self.pathlossB[j] -
                                self.B_Ant_G -
                                self.B_Noise_g))
                #计算A类用户给B类用户的同频干扰,少一个A用户给B用户的路劲损耗
                BB_interference[j] += \
                    10**(0.1* (self.A_power_dB -
                               self.pathlossA_B[i,j] +
                               self.A_Ant_G +
                               self.A_Noise_g))

                for k in range(j+1,len(indexes)):
                    BB_interference[j] += \
                        10**(0.1 * (self.B_power_list[Power_select[indexes[k]]] -
                                    self.pathlossB_B[k,j]-
                                    self.B_Ant_G-
                                    self.B_Noise_g))
                    BB_interference[k] += \
                        10 ** (0.1 * (self.B_power_list[Power_select[indexes[j]]] -
                                      self.pathlossB_B[j, k] -
                                      self.B_Ant_G -
                                      self.B_Noise_g))

            self.B_interference = BB_interference + self.sig2

            #下面根据所得到的signal_power和interference计算通信的比特率
            B_C = np.zeros((self.numB))




