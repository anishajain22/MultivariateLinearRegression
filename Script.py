#!/usr/bin/env python
# coding: utf-8

# In[125]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('seaborn-white')
import sys

my_data = pd.read_csv(sys.argv[1]) #read the data


# In[126]:


#print(my_data)
my_data = (my_data - my_data.mean())/my_data.std()


# In[127]:


#dup.drop(my_data.iloc[:, 1:2])
r_size, c_size = my_data.shape
ratio =0.8

slice_ind=int(ratio*r_size)

X_train = my_data.iloc[0:slice_ind,:-1].values
X_test=my_data.iloc[slice_ind:r_size,:-1]

ones = np.ones([X_train.shape[0],1])
X_train = np.concatenate((ones,X_train),axis=1)

ones = np.ones([X_test.shape[0],1])
X_test = np.concatenate((ones,X_test),axis=1)

Y_train = my_data.iloc[0:slice_ind,c_size-1:c_size].values
Y_test=my_data.iloc[slice_ind:r_size,c_size-1:c_size]


# In[128]:


# print(X_train)
# print(X_test)
# print(Y_train)
# print(Y_test)


# 

# In[129]:


def computeCost(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))


# In[130]:


def computeCostReg(X,y,theta,lmbda):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X)) + np.sum(lmbda*np.power(theta,2))/2


# In[131]:


def rmsError(X,y,theta):
    tobesummed = np.power(((X @ theta.T)-y),2)
    return np.power(np.sum(tobesummed)/len(X),0.5)


# In[132]:


def absError(X,y,theta):
    tobesummed = np.absolute((X @ theta.T)-y)
    return np.sum(tobesummed)/len(X)


# In[133]:


def gradientDescent(X,Y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ (theta.T) - Y), axis=0)
        cost[i] = computeCost(X, Y, theta)
    
    return theta,cost


# In[134]:


def gradientDescentReg(X,Y,theta,iters,alpha,lmbda):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ (theta.T) - Y), axis=0)-lmbda*theta
        cost[i] = computeCostReg(X, Y, theta,lmbda)
    
    return theta,cost


# In[135]:


iters = 100
def trainModelWithoutReg():
    rmsErrors = np.zeros(100)
    absErrors=np.zeros(100)
    alphas=np.zeros(100)
    alpha=0.0
    alphaInc=1/100
    for i in range(100):
        theta = np.zeros([1,c_size])
        alphas[i]=alpha
        alpha+=alphaInc
        theta,cost=gradientDescent(X_train,Y_train,theta,iters,alphas[i])
        rmsErrors[i]=rmsError(X_test,Y_test,theta)
        absErrors[i]=absError(X_test,Y_test,theta)
    fig1, ax1 = plt.subplots()  
    ax1.plot(alphas,rmsErrors,'r')
    ax1.set_xlabel('Alpha')  
    ax1.set_ylabel('Error')  
    ax1.set_title('RMS Error Vs. Alpha without Reg')
    fig1.savefig('fig1.png')
    fig2, ax2 = plt.subplots()  
    ax2.plot(alphas,absErrors,'b')
    ax2.set_xlabel('Alpha')  
    ax2.set_ylabel('Absolute Error')  
    ax2.set_title('Absolute Error Vs. Alpha without Reg')
    fig2.savefig('fig2.png')
trainModelWithoutReg()
    


# In[136]:


iters = 100
def trainModelWithReg():
    rmsErrors = np.zeros(100)
    absErrors=np.zeros(100)
    alphas=np.zeros(100)
    alpha=0.0
    alphaInc=1/100
    lmbda=10
    for i in range(100):
        theta = np.zeros([1,c_size])
        alphas[i]=alpha
        alpha+=alphaInc
        theta,cost=gradientDescentReg(X_train,Y_train,theta,iters,alphas[i],lmbda)
        rmsErrors[i]=rmsError(X_test,Y_test,theta)
        absErrors[i]=absError(X_test,Y_test,theta)
    fig3, ax3 = plt.subplots()  
    ax3.plot(alphas,rmsErrors,'r')
    ax3.set_xlabel('Alpha')  
    ax3.set_ylabel('Error')  
    ax3.set_title('RMS Error Vs. Alpha with Reg')
    fig3.savefig('fig3.png')
    fig4, ax4 = plt.subplots()  
    ax4.plot(alphas,absErrors,'b')
    ax4.set_xlabel('Alpha')  
    ax4.set_ylabel('Absolute Error')  
    ax4.set_title('Absolute Error Vs. Alpha with Reg')
    fig4.savefig('fig4.png')   
trainModelWithReg()
    


# In[137]:


def generateContourWithoutReg():
    theta = np.ones([1,c_size])
        
    th1 = np.linspace(-1000,1000 , num=1000)
    th2 = np.linspace(-1000,1000, num=1000)
    TH1, TH2 = np.meshgrid(th1, th2)
   
    th_copy = theta.copy()
    jv = np.zeros((len(th1), len(th2)))
    for i in range(0,len(th1)):
        th_copy[0][0] = th1[i]
        for j in range(0, len(th2)):
            th_copy[0][1] = th2[j]
            #theta,cost=gradientDescentReg(X_train,Y_train,th_copy,iters,0.5,lmbda)
            jv[i][j] = computeCost(X_train,Y_train,th_copy)
            
    
    fig5,ax5=plt.subplots()
    cp = ax5.contour(TH1, TH2, jv,70,cmap='jet')
    #plt.clabel(cp, inline=True)
    xmin, ymin = np.unravel_index(np.argmin(jv), jv.shape)
    ax5.set_title('Contour Plot without Reg with center ('+str(xmin)+','+str(ymin)+')')
    ax5.set_xlabel('theta1')
    ax5.set_ylabel('theta2')
    fig5.savefig('fig5.png')
    #plt.show()
   
generateContourWithoutReg()

def generateContourWithReg():
    theta = np.ones([1,c_size])
        
    th1 = np.linspace(-1000,1000 , num=1000)
    th2 = np.linspace(-1000,1000, num=1000)
    TH1, TH2 = np.meshgrid(th1, th2)
   
    th_copy = theta.copy()
    jv = np.zeros((len(th1), len(th2)))
    for i in range(0,len(th1)):
        th_copy[0][0] = th1[i]
        for j in range(0, len(th2)):
            th_copy[0][1] = th2[j]
            #theta,cost=gradientDescentReg(X_train,Y_train,th_copy,iters,0.5,lmbda)
            jv[i][j] = computeCostReg(X_train,Y_train,th_copy,10)
            
    
    fig6,ax6=plt.subplots()
    cp = ax6.contour(TH1, TH2, jv,70,cmap='jet')
    xmin, ymin = np.unravel_index(np.argmin(jv), jv.shape)
    ax6.set_title('Contour Plot with Reg with center ('+str(xmin)+','+str(ymin)+')')
    ax6.set_xlabel('theta1')
    ax6.set_ylabel('theta2')
    fig6.savefig('fig6.png')
    #plt.show()
# In[ ]:
generateContourWithReg()




