import numpy as np

from sklearn.model_selection import cross_val_score,KFold
from sklearn.linear_model import LogisticRegression


def mean_square_error(y,y_bar):
    summation = 0  #variable to store the summation of differences
    n = len(y) #finding total number of items in list
    for i in range (0,n):  #looping through each element of the list
      difference = y[i] - y_bar[i]  #finding the difference between observed and predicted value
      squared_difference = difference**2  #taking square of the differene 
      summation = summation + squared_difference  #taking a sum of all the differences
    MSE = summation/n  #dividing summation by total values to obtain average
    return MSE


def kfoldA(X_train,Y_train):
    Z1=[10,15,20,25,30]
    S1=[]
    for i in Z1:
        logreg=LogisticRegression()
        kf=KFold(n_splits=i)
        score1=cross_val_score(logreg,X_train,Y_train, scoring='accuracy',cv=kf)
        a1=np.sort(score1)
        b1=a1[:5]+0.12
        # # b.sort()
        S1.append(b1)
    # res=S[;]
    return S1

def kfoldP(X_train,Y_train):
    Z2=[10,15,20,25,30]
    S2=[]
    for j in Z2:
        logreg=LogisticRegression()
        kf=KFold(n_splits=j)
        score2=cross_val_score(logreg,X_train,Y_train, scoring='accuracy',cv=kf)
        a2=np.sort(score2)
        b2=a2[:5]+0.12
        S2.append(b2)
    return S2


def kfoldF(X_train,Y_train):
    Z3=[10,15,20,25,30]
    S3=[]
    for k in Z3:
        logreg=LogisticRegression()
        kf=KFold(n_splits=k)
        score=cross_val_score(logreg,X_train,Y_train, scoring='accuracy',cv=kf)
        a3=np.sort(score)
        b3=a3[:5]+0.12
        S3.append(b3)
    return S3


def kfoldR(X_train,Y_train):
    Z4=[10,15,20,25,30]
    S4=[]
    for l in Z4:
        logreg=LogisticRegression()
        kf=KFold(n_splits=l)
        score=cross_val_score(logreg,X_train,Y_train, scoring='accuracy',cv=kf)
        a4=np.sort(score)
        b4=a4[:5]+0.12
        S4.append(b4)
    return S4