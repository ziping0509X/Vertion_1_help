import numpy as np

# A = [1,2,3,4,5]
# B = [[4,5,6,7,8],[4,5,6,7,8],[4,5,6,7,8]]
#
#
# A = np.array(A)
# B = np.array(B)
# C = np.append(A, B)
# C.reshape([1,-1])
# print(C)
#
# #C= np.concatenate((A,B),axis= 0)

A = np.ones([3,4,2])
print(A)
print("==========================")
A[1,1] =  100
print(A)
print(A[1][1][0])
B = A.copy()[:,:,0]
print(B)
print(B.shape)