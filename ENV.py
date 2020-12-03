import tensorflow as tf
import numpy as np
import random
import math

NUMA = 5
NUMB = 10
LENTH = 500
HEITH = 500
positionBS = [250,250]

# 12月3号更新版本上传不到GITHUB

class ENVIRONMENT:
    def __init__(self,NUMA,NUMB):
        self.numA = NUMA
        self.numB = NUMB

        self.positionA = self.add_users(self.numA)
        self.positionB = self.add_users(self.numB)
        self.positionA = np.array(self.positionA)
        self.positionB = np.array(self.positionB)

        self.distanceA = self.get_distanceA(self.positionA)
        self.distanceB = self.get_distanceA(self.positionB)
        self.distance_A_B = self.get_distance_A_B(self.positionA,self.positionB)
        self.distance_B_B = self.get_distance_A_B(self.positionB, self.positionB)

        self.pathlossA = self.get_pathlossA(self.positionA, self.distanceA)
        self.pathlossB = self.get_pathlossA(self.positionB, self.distanceB)
        self.pathlossA_B = self.get_pathlossAB(self.distance_A_B)
        self.pathlossB_B = self.get_pathlossBB(self.distance_B_B)

        self.A_power_dB = 23
        self.B_power_list = [10,20,46]
        self.B_power_list = np.array(self.B_power_list)
        self.A_Ant_G = 8
        self.A_Noise_g = 5
        self.B_Ant_G = 3
        self.B_Noise_g = 9

        self.sig2_dB = -114
        self.sig2 = 10 ** (self.sig2_dB / 10)
        #得到了A\B类用户的坐标、相互之间的距离、路径损耗数值

    def add_users(self,n):
        position = np.zeros((n,2))
        for i in range(n):
            ind1 = np.random.randint(0, 500)
            ind2 = np.random.randint(0, 500)

            #start_position = [ind1, ind2]
            position[i,0] = ind1
            position[i,1] = ind2
        return position

    def get_distanceA(self,positionA):
        distanceA = np.zeros(len(positionA))
        for i in range(len(positionA)):
            distance = math.sqrt((positionA[i][0] - 250) ** 2 + (positionA[i][1] - 250) ** 2)
            distanceA[i] = distance
        #print(distanceA)
        return distanceA

    # def get_distanceB(self,positionB):
    #     distanceB = np.zeros([len(positionB), len(positionB)])
    #     for i in range(len(positionB)):
    #         for j in range(len(positionB)):
    #             if not i == j:
    #                 distanceB[i, j] = math.sqrt(
    #                     (positionB[i][0] - positionB[j][0]) ** 2 + (positionB[i][1] - positionB[j][1]) ** 2)
    #     return distanceB

    def get_distance_A_B(self,positionA,positionB):
        distance_A_B = np.zeros((len(positionB),len(positionA)))
        for i in range(len(positionB)):
            for j in range(len(positionA)):
                distance_A_B[i,j] = math.sqrt(
                    (positionA[j,0] - positionB[i,0]) ** 2 + (positionA[j,1] - positionB[i,1]) ** 2
                )
        return distance_A_B

    def get_pathlossA(self,positionA, distanceA):
        PathLoss = np.zeros(len(positionA))
        for i in range(len(positionA)):
            #PathLoss[i] = 128.1 + 37.6 * np.log10(np.sqrt(distanceA[i] ** 2 + (25 - 1.5) ** 2) / 1000)
            PathLoss[i] = 37.6 * np.log10(np.sqrt(distanceA[i] ** 2 + (25 - 1.5) ** 2) / 1000)
            #print(PathLoss_temp)
            #print(PathLoss[i])

        #print("-------------------")
        #print(PathLoss)
        return PathLoss

    def get_pathlossB(self,positionB, distanceB): #先不用这个互相的pathloss，因为假设B类用户也是向基站通信的
        PathLoss = np.zeros(shape=(len(positionB), len(positionB)))

        for i in range(len(positionB)):
            for j in range(len(positionB)):
                #PathLoss[i,j] = 128.1 + 37.6 * np.log10(math.sqrt(distanceB[i,j] ** 2 + (25 - 1.5) ** 2) / 1000)
                PathLoss[i, j] = 37.6 * np.log10(math.sqrt(distanceB[i, j] ** 2 + (25 - 1.5) ** 2) / 1000)

        return PathLoss

    def get_pathlossAB(self,distance_A_B):
        PathLoss = np.zeros((self.numB,self.numA)) #10*5

        for i in range (self.numB):
            for j in range(self.numA):
                #PathLoss[i,j] = 128.1 + 37.6 * np.log10(math.sqrt(distance_A_B[i,j] ** 2 + (25 - 1.5) ** 2) / 1000)
                PathLoss[i, j] = 37.6 * np.log10(math.sqrt(distance_A_B[i, j] ** 2 + (25 - 1.5) ** 2) / 1000)

        return PathLoss

    def get_pathlossBB(self, distance_A_B):
        PathLoss = np.zeros((self.numB, self.numB))

        for i in range(self.numB):
            for j in range(self.numB):
                #PathLoss[i, j] = 128.1 + 37.6 * np.log10(math.sqrt(distance_A_B[i,j] ** 2 + (25 - 1.5) ** 2) / 1000)
                PathLoss[i, j] = 37.6 * np.log10(math.sqrt(distance_A_B[i, j] ** 2 + (25 - 1.5) ** 2) / 1000)

        return PathLoss

    def get_state(self):
        #self.state_input = []
        self.pathlossA = np.array(self.pathlossA)
        self.pathlossB = np.array(self.pathlossB)
        self.state_input = np.append(self.pathlossA, self.pathlossB)
        self.state_input = self.state_input.reshape((1,15))

        return self.state_input

    def get_reward(self,stateinput,actionall,idx): #单词循环中给单用户选择动作得到的奖励，如何设置？
        #action输入的是一个数字，5*3 = 15
        print("开始调试get_reward================================")
        actionall_1 = actionall.copy()
        RB_select = actionall[:,0] #[1选择的RB，2选择的RB，......,10选择的RB]
        print("RB_select:")
        print(RB_select)
        Power_select = actionall[:,1] #[1选择的P，2选择的P，......,10选择的P]
        print("Power_select:")
        print(Power_select)
        B_signal = np.zeros(self.numB)
        BB_interference = np.zeros((self.numB))
        #去掉自己，方便后面大循环
        #RB_select[idx] = 100
        #下面开始计算B_signal、BB_interference

        for i in range(self.numB): #i指示的是B类用户的序号

            indexes = np.argwhere(RB_select == i) #返回了一个一维数组,里面的元素也应该就是整数
            indexes_temp = np.zeros(len(indexes),dtype=int)
            #print(indexes)

            for l in range(len(indexes)):
                indexes_temp[l] = indexes[l][0]
                #print(indexes[l][0])
                #print(indexes_temp)
            #print("indexes_temp is:")
            #print(indexes_temp)
            #print(indexes_temp)
            indexes = indexes_temp.copy()
            #print("indexes is:")
            #print(indexes)

            for j in range(len(indexes)): #j和index[j]指示的是和i号B类用户产生干扰的用户
                #print(indexes[j,0])
                #print("here")
                #print(indexes[j])
                B_signal[ indexes[j] ] = \
                    10**(0.1 * (self.B_power_list[ Power_select[indexes[j]] ] - self.pathlossB[indexes[j]] - self.B_Ant_G -self.B_Noise_g))
                print("B_signal[ indexes[j] ] is:")
                print(B_signal[ indexes[j] ])
                #计算A类用户给B类用户的同频干扰
                #pathloss_A_B 是一个10*5的数组
                # BB_interference[indexes[j]] += \
                #     10**(0.1* (self.A_power_dB -
                #                self.pathlossA_B[indexes[j],i] +
                #                self.A_Ant_G +
                #                self.A_Noise_g))

                BB_interference[indexes[j]] += \
                    10 ** (0.1 * (self.A_power_dB -
                                  self.A_Ant_G +
                                  self.A_Noise_g))
                print(BB_interference[indexes[j]])

                #计算B类用户给B类用户的同频干扰
                for k in range(j+1,len(indexes)):
                    BB_interference[indexes[j]] += \
                        10**(0.1 * (self.B_power_list[Power_select[indexes[k]]] -
                                    self.pathlossB_B[indexes[k],indexes[j]]-
                                    self.B_Ant_G-
                                    self.B_Noise_g))
                    BB_interference[indexes[k]] += \
                        10 ** (0.1 * (self.B_power_list[Power_select[indexes[j]]] -
                                      self.pathlossB_B[indexes[j], indexes[k]] -
                                      self.B_Ant_G -
                                      self.B_Noise_g))

        self.B_interference = BB_interference + self.sig2
        print(B_signal)
        print(self.B_interference)

        #下面根据所得到的signal_power和interference计算通信的比特率
        B_C = np.zeros((self.numB))
        for i in range(len(B_C)):
            B_C[i] = np.log2(1 + B_signal[i] / (BB_interference[i] + self.sig2))
        print("B_C is:")
        print(B_C)

        B_C_SUM = 0
        for i in range(len(B_C)):
            B_C_SUM += B_C[i]

        AB_interference = np.zeros(self.numA)
        A_signal = np.zeros(self.numA)
        for i in range(self.numA):
            A_signal[i] = 10**(0.1*(self.A_power_dB - self.pathlossA[i] + self.A_Ant_G - self.A_Noise_g))
        #下面开始计算A类用户的通信比特率
        for i in range(self.numA):
            indexes = np.argwhere(RB_select == i)
            for j in range(len(indexes)):
                #这里先使用一个简化的用法
                AB_interference[i] += 10**(0.1*(Power_select[indexes[j]] - self.pathlossA_B[j][i]))

        AB_interference = AB_interference+ self.sig2

        print("*****************")

        print(A_signal)
        print(AB_interference)

        A_C = np.zeros(self.numA)
        for i in range(self.numA):
            A_C[i] = np.log10(1 + A_signal[i] / AB_interference[i])

        print("A_C is:")
        print(A_C)
        A_C_SUM = 0
        for i in range(len(A_C)):
            A_C_SUM += A_C[i]

        lamd = 0.8
        reward = A_C_SUM + lamd * B_C_SUM

        print("reward is:")
        #print(A_C_SUM)
        #print(B_C_SUM)
        print(reward)

        return reward


























