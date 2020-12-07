import tensorflow as tf
import numpy as np
import random
import math
from ENV import ENVIRONMENT
import  matplotlib.pyplot as plt
from DQN import Qnetwork
import pandas as pd

#============================
# 二维平面是500*500
# 生成A、B类用户的坐标、相互之间的距离
# 根据以上数据，生成path_loss\shadow\fastfading的数据(考虑只用path_loss)
# 是坐标变动以后，仍然有学习效果？还是只能在指定坐标下面一步步优化到max
# 确定state_input、get_reward（11月30日15点50分）
# 开始使用自己的的架构写这个程序（12月2日15点01分）
# 开始对version0.9调参（12月4号9点02分）V0.915
#============================
NUMA = 5
NUMRB = NUMA
NUMB = 10
NUMPOWER = 3
LENTH = 500
HEITH = 500
BS_POSITION = [250, 250]

N = 20000
Time = 0

Qnetwork = Qnetwork(NUMA,NUMB)
Env = ENVIRONMENT(NUMA,NUMB)

Reward = []
R_total = 0
Loss = []

for Time in range(0,1):
    if Time == 0:
        Reward = []
        R_total = 0
        Loss = []
    for i in range(NUMB):

        print("**********************************************")
        print("the iterations_small %d of iterations_big %d:" %(i,Time))

        state_old = Env.get_state()
        action_array,action= Qnetwork.getAction(actionNum=NUMA*3,stateInput=state_old)
        Qnetwork.action_all[i, 1] = action % NUMPOWER
        Qnetwork.action_all[i, 0] = int(np.floor(action / NUMPOWER))

        print("select RB is:")
        print(Qnetwork.action_all[i, 0])
        print("select Power is:")
        print(Qnetwork.action_all[i, 1])
        print("Qnetwork-action_all is:")
        print(Qnetwork.action_all)

        reward = Env.get_reward(stateinput=state_old,actionall=Qnetwork.action_all,idx=i)
        state_new = Env.get_state()

        loss = Qnetwork.getLoss(currentState=state_old,nextState=state_new,action=action_array,reward=reward)
        R_total += reward
        Reward.append(R_total)

        if not loss == 0:
            Loss.append(loss)


plt.plot(Loss)
plt.show()

# Loss1 = np.array(Loss)
# data1 = pd.DataFrame(Loss1,columns=['Loss'])
# data1.to_csv("D:\YuanZihong\SensorModel\Loss.csv")
#
# Reward1 = np.array(Reward)
# data1 = pd.DataFrame(Reward1,columns=['Reward'])
# data1.to_csv("D:\YuanZihong\SensorModel\Reward.csv")
#
# plt.rcParams["font.family"]="SimHei"
# plt.rcParams['axes.unicode_minus']=False
#
# fig = plt.figure(num= 1,figsize=(12,6))
# ax1 = fig.add_subplot(121)
# ax1.set_xlim(0,36)
# ax1.set_xlabel("时间/t")
# ax1.set_ylabel("函数值")
# ax1.set_title("环境状态集[0-36]")
# ax1.scatter(t1,sig,c= 'b')
#
# fig = plt.figure(num= 1,figsize=(12,6))
# ax1 = fig.add_subplot(122)
# ax1.set_xlim(0,8)
# ax1.set_xlabel("num")
# ax1.set_ylabel("choice")
# ax1.set_title("ActionChoice[0,9]")
# ax1.scatter(t2,actionChoice,c= 'y')
#
# fig = plt.figure(num= 2,figsize=(12,6))
# ax1 = fig.add_subplot(121)
# ax1.set_xlabel("迭代次数")
# ax1.set_ylabel("损失函数值")
# ax1.set_title("损失函数曲线")
# ax1.plot(Loss)
#
# fig = plt.figure(num= 2,figsiz=(12,6))
# ax1 = fig.add_subplot(122)
# ax1.set_xlabel("迭代次数")
# ax1.set_ylabel("总奖励")
# ax1.set_title("奖励函数曲线")
# ax1.plot(Reward)
#
# fig = plt.figure(num=3,figsize =(12,6))
# ax1 = fig.add_subplot(111)
# ax1.set_xlabel("时间/t")
# ax1.set_ylabel("动作值")
# ax1.set_title("经过训练以后的成果")
# ax1.set_xlim(199900,199928)
# ax1.scatter(t1,sig,c='b',label="输入信号")
# lenth = len(ActionShow)
# t3 = np.linspace(0,lenth,num=lenth)
# ax1.scatter(t3,ActionShow,c='y',label="输出动作")
# ax1.legend(loc=1)
#
# plt.show()

























