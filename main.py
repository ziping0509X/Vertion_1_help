import tensorflow as tf
import numpy as np
import random
import math
from ENV import ENVIRONMENT
import  matplotlib.pyplot as plt
from DQN import Qnetwork

NUMA = 5
NUMB = 10
LENTH = 500
HEITH = 500

N = 20000

Qnetwork = Qnetwork(NUMA,NUMB)
Env = ENVIRONMENT(NUMA,NUMB)

Reward = []
R_total = 0
Loss = []


