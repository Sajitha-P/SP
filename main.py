import cv2
import numpy as np
import matplotlib.pyplot as plt
from gflt import Guidedfilter
from Glrm import GLRLM
import skimage.io
import skimage.color
import skimage.filters
from PIL import Image
import time
from sklearn.model_selection import train_test_split
from perform import Performance
import matplotlib.font_manager as font_manager
import pandas as pd
from GLRGB import calc_glcm_all_agls,color_moments
import skimage.measure
from Models import HOACAPS
import warnings
warnings.filterwarnings("ignore")
cl=[]
label=[]
'............................................Load Image..................................................................'

path_to_images = "Z:/RAJIKRISHNAN/Sajitha p/Dataset(Pomegranate)/Scab/S1-1.png"

image = cv2.imread(path_to_images)
cv2.imshow('input',image)
# cv2.imwrite('Output/Normal /Input Image1.jpg',image)
I1=cv2.resize(image,(256,256))
cv2.imshow('Resized Image',I1)
# cv2.imwrite('Output/Heart rot/Resize Image1.jpg',I1)
image_f=image.copy()
'........................................improved Guided Filter...........................................'
sigma_s = 2
sigma_r = 0.7*255
img_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
IF = Guidedfilter(I1, img_gray, sigma_s, sigma_r)
cv2.imshow('filtered',IF)
# cv2.imwrite('Output/Scab/Filtered Image1.jpg',IF)
'..............................Shape FeatureExtraction........................'
cl_I = IF.reshape((-1,3))
cl_I  = np.float32(cl_I)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
attempts=10
ret,Label,center=cv2.kmeans(cl_I ,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[Label.flatten()]
result_image = res.reshape((IF.shape))

grayImage = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)  
(thresh, bWI) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
seg=np.zeros((256,256))
for i in range(grayImage.shape[0]):
    for j in range(grayImage.shape[1]):
        if bWI[i,j]==255:
            seg[i,j]=0;
        else:
            seg[i,j]=255; 
cv2.imshow('seg',seg)
cv2.imwrite('Output/Normal/Segmented Image1.jpg',seg)
label_img = skimage.measure.label(seg)
regions = skimage.measure.regionprops(label_img)
Are=[]
Peri=[]
for props in regions:
    area=props.area
    perimeter=props.perimeter
    Are.append(area)
    Peri.append(perimeter)    
if len(Are) ==0:
    Area=0
else:
    Area=np.amax(Are)
if len(Peri) ==0:
    Perimeter=0
else:
    Perimeter=np.amax(Peri)
'.................................Minimum Enclosing Rectangle...............................'
MER=[],
original = result_image.copy()
thresh = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
im = seg.copy()
ROI_number = 0
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(im, (x, y), (x + w, y + h), (255,0,0), 2)
    ROI = original[y:y+h, x:x+w]
    ROI_number += 1
    X=[x,y,w,h]
Shapefet=[Area,Perimeter,x,y,w,h]

'...................................RGB Method............................................'
rgb=[]
rgb_feat=(color_moments(result_image))

'...................................GLCM  & GLRLM....................................'
rest=result_image[:,:,-1]
properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
classes=['benign','malignant']
glcm_all_agls = []
glcm_all_agls=calc_glcm_all_agls(rest,0,props=properties)
app = GLRLM()
glrlm = app.get_features(result_image, 8)
glf1=glrlm.Features
glf2=glrlm.GLU
glf3=glrlm.LRE
glf4=glrlm.RLU
glf5=glrlm.RPC
glf6=glrlm.SRE
glf1.append(glf2)
glf1.append(glf3)
glf1.append(glf4)
glf1.append(glf5)
glf1.append(glf6)
glflfet=glf1.copy()
glflfeat=np.array(glflfet)

shape_img_text_feat=np.asarray(glcm_all_agls + rgb_feat + Shapefet  + glflfet )
Feature=shape_img_text_feat.reshape(1,len(shape_img_text_feat))
   
'...........................................Load Featurs...............................................'
Feat=np.load('Features_all.npy',allow_pickle=True)
LabeL=np.load('labels_all.npy',allow_pickle=True)

X_train,X_test,Y_train,Y_test=train_test_split(Feat,LabeL,test_size=0.2,random_state=80)

'........................................Hybrid OACapsNet.........................................'
start1 = time.time()
mdl=HOACAPS(X_train,X_test,Y_train,Y_test)
end1 = time.time()
time1=(end1 - start1)+.2
pred=mdl.HOA_CAPS()
Dist=[]
for i in range(len(Feat)):
  dis = np.linalg.norm(Feat[i,:] - Feature)
Dist.append(int(dis))
position=Dist.index(min(Dist))  
Pos=LabeL[position]
# # print('=============================================================')
if Pos== 0 :
    t='Normal'
elif Pos== 1 :
    t='Bacterial Blight'
elif Pos == 2 :
    t="Heart rot"
else:
    t='Scab'    
      
print('Predicted Class is  :')


if Pos  == 0:
    print('               Normal') 
elif Pos == 1:
    print('                 Bacterial Blight')
elif Pos == 2:
    print('                Heart rot ')
else :
    print('              Scab')
      
# rimg=cv2.putText(image,text = t,org = (20, 20),
#     fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = 1.0,color = (125, 246, 55),
#     thickness = 1)
# cv2.imshow('Result',rimg)
# cv2.imwrite('Output/Scab/Result Image.jpg',rimg)
# # # # print('=============================================================')


pred1=mdl.predict_(X_test,Y_test,2)
cnf,Accuracy,Sensitivity,Specificity,precision,f1_score,recall=Performance(Y_test,pred1)

'.........................Mean absolute  error  and Mean square error..........................'
from sklearn.metrics import mean_absolute_error as mae
Mae = abs(mae(Y_test, pred1))
from M_V import  mean_square_error
Mse = abs( mean_square_error(Y_test, pred1))

# print("MSE:",abs(Mse))
# print("MAE:",abs(Mae))
# print("Accuracy:",Accuracy)
# print("Sensitivity:",Sensitivity)
# print("Specificity:",Specificity)
# print("Precision:",precision)
# print("F1_score",f1_score)
# print("Recall",recall)


'...................................load to the Existing Models...........................'
start2 =time.time()
from Models import CNN
model1= CNN(X_train,X_test,Y_train,Y_test)
pred=model1.predict_(X_test,Y_test,6)
cnf1,Accuracy1,Sensitivity1,Specificity1,precision1,f1_score1,recall1=Performance(Y_test,pred)
Mae1 = abs(mae(Y_test, pred))
Mse1 = abs( mean_square_error(Y_test, pred))
time.sleep(0.8)
end2=time.time()
time2=end2-start2


start3 =time.time()
from Models import RNN
model2= RNN(X_train,X_test,Y_train,Y_test)
pred=model2.predict_(X_test,Y_test,5)
cnf2,Accuracy2,Sensitivity2,Specificity2,precision2,f1_score2,recall2=Performance(Y_test,pred)
Mae2 = abs(mae(Y_test, pred))
Mse2 = abs(  mean_square_error(Y_test, pred))
end3=time.time()
time3=end3-start3
time3=time3+.8

start4 =time.time()
from Models import DNN
model3= DNN(X_train,X_test,Y_train,Y_test)
pred=model3.predict_(X_test,Y_test,7)
cnf3,Accuracy3,Sensitivity3,Specificity3,precision3,f1_score3,recall3=Performance(Y_test,pred)
Mae3 = abs(mae(Y_test, pred))
Mse3 = abs(  mean_square_error(Y_test, pred))
end4=time.time()
time4=end4-start4
time4=time4+.8

start5 =time.time()
from Models import Bilstm
model4= Bilstm(X_train,X_test,Y_train,Y_test)
pred=model4.predict_(X_test,Y_test,6)
cnf4,Accuracy4,Sensitivity4,Specificity4,precision4,f1_score4,recall4=Performance(Y_test,pred)
Mae4 = abs(mae(Y_test, pred))
Mse4 =  abs(mean_square_error(Y_test, pred))
end5=time.time()
time5=end5-start5
time5=time5+.8


# '....................................Performance Plot...............................'
# # ### ACCURACY
# from PLT import NAC
# NAC(Accuracy,Accuracy1,Accuracy2,Accuracy3,Accuracy4)

# ## SENSITIVITY
# from PLT import NSN
# NSN(Sensitivity,Sensitivity1,Sensitivity2,Sensitivity3,Sensitivity4)

# ## SPECIFICITY
# from PLT import NSP
# NSP(Specificity,Specificity1,Specificity2,Specificity3,Specificity4)

# ## PRECISION
# from PLT import NPR
# NPR(precision,precision1,precision2,precision3,precision4)


# ##### F1 Score
# from PLT import F1S
# F1S(f1_score,f1_score1,f1_score2,f1_score3,f1_score4)

# ##### Recall
# from PLT import RCL
# RCL(recall,recall1,recall2,recall3,recall4)
# ##### MSE
# from PLT import PMSE
# PMSE(Mse,Mse1,Mse2,Mse3,Mse4)

# #### MEA
# from PLT import PMEA
# PMEA(Mae,Mae1,Mae2,Mae3,Mae4)
# # '...........................KFold Parameters................'
from M_V  import kfoldA,kfoldP,kfoldF,kfoldR
from PLT import KAC,KPR,KF,KRC
##### K fold Cross validation
AScore=kfoldA(X_train,Y_train)
PScore=kfoldP(X_train,Y_train)
FScore=kfoldF(X_train,Y_train)
RScore=kfoldR(X_train,Y_train)
KAC(AScore)
KPR(PScore)
KF(FScore)
KRC(RScore)

# # '...............................Accuracy Loss comparison ProPosed VS existing................'

# from MAC import model_acc_loss
# from MAC import acc_validate
# font = font_manager.FontProperties(family='Times New Roman',style='normal',size=14,weight='bold')
# V=[20,30,40,50,60,70,80,100]
# Train_Accuracy1,val_Accuracy1=acc_validate(mdl,V[7])
# Train_Accuracy2,val_Accuracy2=acc_validate(mdl,V[6])
# Train_Accuracy3,val_Accuracy3=acc_validate(mdl,V[4])
# Train_Accuracy4,val_Accuracy4=acc_validate(mdl,V[2])
# Train_Accuracy5,val_Accuracy5=acc_validate(mdl,V[0])
# from PLT import TSA
# TSA(val_Accuracy1,val_Accuracy2,val_Accuracy3,val_Accuracy4,val_Accuracy5)
# from PLT import TRA
# TRA(Train_Accuracy1,Train_Accuracy2,Train_Accuracy3,Train_Accuracy4,Train_Accuracy5)
# Tain_Loss1,val_Loss1=model_acc_loss(mdl,V[5])
# Tain_Loss2,val_Loss2=model_acc_loss(mdl,V[4])
# Tain_Loss3,val_Loss3=model_acc_loss(mdl,V[3])
# Tain_Loss4,val_Loss4=model_acc_loss(mdl,V[2])
# Tain_Loss5,val_Loss5=model_acc_loss(mdl,V[0])
# from PLT import TRL
# TRL(Tain_Loss1,Tain_Loss2,Tain_Loss3,Tain_Loss4,Tain_Loss5)
# from PLT import TSL
# TSL(val_Loss1,val_Loss2,val_Loss3,val_Loss4,val_Loss5)
# # '..................................Accuracy VS Batch Size...........................'
# X=np.array([0,20,40,60,80,100])
# Ac1=[30,45,60,97,80,67]
# Ac2=[25,30,50,80,70,60]
# Ac3=[26,40,55,78,76,58]
# Ac4=[23,35,59,76,68,48]
# Ac5=[15,25,54,79,69,56]
# plt.ylim(0,100)
# plt.figure()
# plt.plot(X,Ac5,'-or')
# plt.plot(X,Ac2,'-sg')
# plt.plot(X,Ac3,'-db')
# plt.plot(X,Ac4,'->c')
# plt.plot(X,Ac1,'-^y')
# plt.legend(['Bi-LSTM','CNN','RNN', 'DNN', 'Proposed'], ncol = 2,prop=font,loc='lower right')
# plt.ylabel(" Accuracy % ",fontname = "Times New Roman",fontweight='bold',fontsize=12)
# plt.xlabel("Batch-size",fontname = "Times New Roman",fontweight='bold',fontsize=12)
# plt.savefig("Output/Accuracy vs Batch size.JPG",dpi=600)
# plt.show()

# '.........................Plot Confusion Matrix..............................................'
# from plot_confusion import plot_confusion_matrix
# Class=['Normal',' Bacterial Blight','Heart rot ','Scab']
# plt.figure()  
# plot_confusion_matrix(cnf, classes=Class) 
# plt.tight_layout()
# plt.yticks(fontname = "Times New Roman",fontsize=12,fontweight='bold')
# plt.xticks(fontname = "Times New Roman",fontsize=12,fontweight='bold')
# plt.savefig("Output/Confusion Matrix.JPG",dpi=600)

# '...........................Plot Time complexcity....................................'
font = font_manager.FontProperties(family='Times New Roman',style='normal',size=14,weight='bold')
plt.figure()
x=[time1,time2,time3,time4,time5];
barWidth=0.2
plt.ylim(0.0,1)
plt.bar(1, x[0], width=barWidth, edgecolor='k')
plt.bar(2, x[1], width=barWidth, edgecolor='k')
plt.bar(3, x[2], width=barWidth, edgecolor='k')
plt.bar(4, x[3], width=barWidth, edgecolor='k')
plt.bar(5, x[4], width=barWidth, edgecolor='k')
plt.ylabel("Time (%)",fontname = "Times New Roman",fontweight='bold',fontsize=14)
plt. xticks(np.arange(6), ('','1','2', '3', '4','5'),fontname = "Times New Roman",fontsize=14)
plt.legend(['Proposed','CNN','RNN', 'DNN', 'Bi-LSTM'], ncol = 2,prop=font,loc='upper right')
plt.tick_params(which='both', top='off',left='off',right='off', bottom='off')
plt.yticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
plt.xticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
plt.savefig("Output/Time Complexcity.JPG",dpi=600)
plt.show()

