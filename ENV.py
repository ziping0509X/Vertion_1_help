import tensorflow as tf
import numpy as np
import random
import math

NUMA = 5
NUMB = 10
LENTH = 1000
HEITH = 1000
positionBS = [500,500]

# 12月3号更新版本上传不到GITHUB  V0.915
# 修改get_pathloss的式子
# L(dB) = 0.837 + 20log10(d/10)

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

        self.B_signal = np.zeros(self.numB)
        self.BB_interference = np.zeros((self.numB))

        self.AB_interference = np.zeros(self.numA)
        self.A_signal = np.zeros(self.numA)

        self.A_power_dB = 23
        self.B_power_list = [5,10,23]
        self.B_power_list = np.array(self.B_power_list)
        self.A_Ant_G = 10
        self.A_Noise_g = 3
        self.B_Ant_G = 5
        self.B_Noise_g = 3

        self.sig2_dB = -114
        self.sig2 = 10 ** (self.sig2_dB / 10)
        #得到了A\B类用户的坐标、相互之间的距离、路径损耗数值

    def get_A_ini_C_SUM(self):
        A_SUM = 0
        A_C = np.zeros(len(self.positionA))
        for i in range(len(self.positionA)):
            A_SING = 10**(0.1*(self.A_power_dB - self.pathlossA[i] + self.A_Ant_G - self.A_Noise_g))
            A_C[i] = 150000 * np.log10(1 + A_SING / self.sig2)
            A_SUM += 150000 * np.log10(1 + A_SING / self.sig2)
        return A_C,A_SUM


    def add_users(self,n):
        position = np.zeros((n,2))
        for i in range(n):
            ind1 = np.random.randint(0, 1000)
            ind2 = np.random.randint(0, 1000)

            #start_position = [ind1, ind2]
            position[i,0] = ind1
            position[i,1] = ind2
        return position

    def get_distanceA(self,positionA):
        distanceA = np.zeros(len(positionA))
        for i in range(len(positionA)):
            distance = math.sqrt((positionA[i][0] - 500) ** 2 + (positionA[i][1] - 500) ** 2)
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
            PathLoss[i] = 0.837 + 20 * np.log10(distanceA[i] / 10)
            #PathLoss[i] = 37.6 * np.log10(np.sqrt(distanceA[i] ** 2 + (25 - 1.5) ** 2) / 1000)
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
                #PathLoss[i, j] = 37.6 * np.log10(math.sqrt(distanceB[i, j] ** 2 + (25 - 1.5) ** 2) / 1000)
                PathLoss[i,j] = 0.837 + 20 * np.log10(distanceB[i,j] / 10)

        return PathLoss

    def get_pathlossAB(self,distance_A_B):
        PathLoss = np.zeros((self.numB,self.numA)) #10*5

        for i in range (self.numB):
            for j in range(self.numA):
                #PathLoss[i,j] = 128.1 + 37.6 * np.log10(math.sqrt(distance_A_B[i,j] ** 2 + (25 - 1.5) ** 2) / 1000)
                #PathLoss[i, j] = 37.6 * np.log10(math.sqrt(distance_A_B[i, j] ** 2 + (25 - 1.5) ** 2) / 1000)
                PathLoss[i, j] = 0.837 + 20 * np.log10(distance_A_B[i, j] / 10)

        return PathLoss

    def get_pathlossBB(self, distance_A_B):
        PathLoss = np.zeros((self.numB, self.numB))

        for i in range(self.numB):
            for j in range(self.numB):
                #PathLoss[i, j] = 128.1 + 37.6 * np.log10(math.sqrt(distance_A_B[i,j] ** 2 + (25 - 1.5) ** 2) / 1000)
                #PathLoss[i, j] = 37.6 * np.log10(math.sqrt(distance_A_B[i, j] ** 2 + (25 - 1.5) ** 2) / 1000)
                PathLoss[i, j] = 0.837 + 20 * np.log10(distance_A_B[i, j] / 10)

        return PathLoss

    def get_state(self):
        #self.state_input = []
        #环境状态到底有没有改变？这里的环境状态肯定是改变了！
        self.pathlossA = np.array(self.pathlossA)
        self.pathlossB = np.array(self.pathlossB)
        self.state_input = np.append(self.pathlossA, self.pathlossB)
        self.state_input = self.state_input.reshape((1,15))

        return self.state_input

    def get_state_2(self,index):
        #环境一直是在改变的，并不是一个一成不变的内容，由于之前把pathloss输入、把index输入，都没有准确地表征环境，所以模型根本没有找到映射。
        state = np.zeros(35)
        for i in range(5):
            state[i] = index
        for i in range(5,15):
            state[i] = self.B_signal[i-5]
        for i in range(15,25):
            state[i] = self.BB_interference[i-15]
        for i in range(25,30):
            state[i] = self.A_signal[i-25]
        for i in range(30,35):
            state[i] = self.AB_interference[i-30]

        state = state.reshape((1,35))

        return state



    def get_state_1(self,index):
        self.state_input = np.zeros(15)
        for i in range(15):
            self.state_input[i] = index
        self.state_input = self.state_input.reshape((1, 15))
        return self.state_input

    # def get_reward(self,stateinput,actionall,idx): #单次循环中给单用户选择动作得到的奖励，如何设置？
    #     #action输入的是一个数字，5*3 = 15
    #     actionall_1 = actionall.copy()
    #     RB_select = actionall[:,0] #[1选择的RB，2选择的RB，......,10选择的RB]
    #     Power_select = actionall[:,1] #[1选择的P，2选择的P，......,10选择的P]
    #
    #     for i in range(self.numB): #i指示的是B类用户的序号
    #
    #         indexes = np.argwhere(RB_select == i) #返回了一个一维数组,里面的元素也应该就是整数
    #         indexes_temp = np.zeros(len(indexes),dtype=int)
    #         #print(indexes)
    #
    #         for l in range(len(indexes)):
    #             indexes_temp[l] = indexes[l][0]
    #         indexes = indexes_temp.copy()
    #         #print("indexes is:")
    #         #print(indexes)
    #
    #         for j in range(len(indexes)): #j和index[j]指示的是和i号B类用户产生干扰的用户
    #             #print(indexes[j,0])
    #             #print("here")
    #             #print(indexes[j])
    #             self.B_signal[ indexes[j] ] = \
    #                 10**(0.1 * (self.B_power_list[ Power_select[indexes[j]] ] - self.pathlossB[indexes[j]] + self.B_Ant_G -self.B_Noise_g))
    #             #print("B_signal[ indexes[j] ] is:")
    #             #print(B_signal[ indexes[j] ])
    #             #计算A类用户给B类用户的同频干扰
    #             #pathloss_A_B 是一个10*5的数组
    #             # BB_interference[indexes[j]] += \
    #             #     10**(0.1* (self.A_power_dB -
    #             #                self.pathlossA_B[indexes[j],i] +
    #             #                self.A_Ant_G +
    #             #                self.A_Noise_g))
    #
    #             #这种算法导致A类用户所有的功率都投射在了B类用户身上
    #             self.BB_interference[indexes[j]] = \
    #                 10 ** (0.1 * (self.A_power_dB +
    #                               self.A_Ant_G - self.pathlossA_B[indexes[j],i] -
    #                               self.A_Noise_g))
    #             #print(BB_interference[indexes[j]])
    #
    #             #计算B类用户给B类用户的同频干扰
    #             for k in range(j+1,len(indexes)):
    #                 self.BB_interference[indexes[j]] += \
    #                     10**(0.1 * (self.B_power_list[Power_select[indexes[k]]] -
    #                                 self.pathlossB_B[indexes[k],indexes[j]]-
    #                                 self.B_Ant_G-
    #                                 self.B_Noise_g))
    #                 self.BB_interference[indexes[k]] += \
    #                     10 ** (0.1 * (self.B_power_list[Power_select[indexes[j]]] -
    #                                   self.pathlossB_B[indexes[j], indexes[k]] -
    #                                   self.B_Ant_G -
    #                                   self.B_Noise_g))
    #
    #     self.B_interference = self.BB_interference + self.sig2
    #
    #     #下面根据所得到的signal_power和interference计算通信的比特率
    #     B_C = np.zeros((self.numB))
    #     for i in range(len(B_C)):
    #         B_C[i] =  150*np.log2(1 + self.B_signal[i] / (self.B_interference[i] + self.sig2))
    #
    #     B_C_SUM = 0
    #     for i in range(len(B_C)):
    #         B_C_SUM += B_C[i]
    #
    #     self.AB_interference = np.zeros(self.numA)
    #     self.A_signal = np.zeros(self.numA)
    #     for i in range(self.numA):
    #         self.A_signal[i] = 10**(0.1*(self.A_power_dB - self.pathlossA[i] + self.A_Ant_G - self.A_Noise_g))
    #     #下面开始计算A类用户的通信比特率
    #     for i in range(self.numA):
    #         indexes = np.argwhere(RB_select == i)
    #         for j in range(len(indexes)):
    #             #这里先使用一个简化的用法
    #             self.AB_interference[i] += 10**(0.1*(Power_select[indexes[j]] - self.pathlossA_B[j][i]))
    #
    #     AB_interference = self.AB_interference+ self.sig2
    #     A_C = np.zeros(self.numA)
    #     for i in range(self.numA):
    #         A_C[i] = np.log10(1 + self.A_signal[i] / AB_interference[i])
    #
    #     print("A_C is:")
    #     print(A_C)
    #     print("B_C is:")
    #     print(B_C)
    #
    #     A_C_SUM = 0
    #     for i in range(len(A_C)):
    #         A_C_SUM += A_C[i]
    #
    #     lamd = 1.5
    #     reward = A_C_SUM + lamd * B_C_SUM
    #
    #     print("A_C_SUM and B_C_SUM is:%d bit/s, %d bit/s"%(A_C_SUM,B_C_SUM))
    #     reward = reward
    #
    #     print("reward is %d 00kbit/s:"%reward)
    #
    #     return reward

    def get_reward(self, actionall):

        RB_selecet = actionall[:, 0]
        Power_select = actionall[:, 1]
        self.A_signal = np.zeros(self.numA)
        self.B_signal = np.zeros(self.numB)

        for i in range(self.numA):  # 一次只处理一个用户
            self.A_signal[i] = 10 ** (0.1 * (self.A_power_dB - self.pathlossA[i] + self.A_Ant_G - self.A_Noise_g))

            indexes = np.argwhere(RB_selecet == i)
            for j in range(len(indexes)):
                if not indexes[j] == i:
                    self.AB_interference[i] += 10 ** (0.1 * (Power_select[indexes[j]] - self.pathlossA_B[j][i]))

        AB_interference = self.AB_interference + self.sig2
        A_C = np.zeros(self.numA)

        for i in range(self.numA):
            A_C[i] = np.log10(1 + self.A_signal[i] / AB_interference[i])

        A_C_SUM = 0
        for i in range(len(A_C)):
            A_C_SUM += A_C[i]

        for i in range(self.numB):
            self.B_signal[i] = \
                10 ** (0.1 * (self.B_power_list[Power_select[i]] - self.pathlossB[
                    i] + self.B_Ant_G - self.B_Noise_g))

            self.BB_interference[i] =   10 ** (0.1 * (self.A_power_dB +
                              self.A_Ant_G - self.pathlossA_B[i,RB_selecet[i]] -
                              self.A_Noise_g))

            indexes = np.argwhere(RB_selecet == i)  # 返回了一个一维数组,里面的元素也应该就是整数
            indexes_temp = np.zeros(len(indexes), dtype=int)
            # print(indexes)

            for l in range(len(indexes)):
                indexes_temp[l] = indexes[l][0]
            indexes = indexes_temp.copy()

            for j in range(len(indexes)):
                if not indexes[j] == i:
                    self.BB_interference[i] += \
                        10 ** (0.1 * (self.B_power_list[Power_select[indexes[j]]] -
                                      self.pathlossB_B[i, indexes[j]] +
                                      self.B_Ant_G -
                                      self.B_Noise_g))

        self.B_interference = self.BB_interference + self.sig2

        B_C = np.zeros((self.numB))
        for i in range(len(B_C)):
            B_C[i] = 150 * np.log2(1 + self.B_signal[i] / (self.B_interference[i] + self.sig2))

        B_C_SUM = 0
        for i in range(len(B_C)):
            B_C_SUM += B_C[i]

        lamd = 0.1
        reward = A_C_SUM + lamd * B_C_SUM

        print("A_C_SUM and B_C_SUM is:%d bit/s, %d bit/s" % (A_C_SUM, B_C_SUM))
        reward = reward

        print("reward is %d 00kbit/s:" % reward)

        return reward