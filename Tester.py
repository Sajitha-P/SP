# import cv2
# import numpy as np



# while True:
#     img=cv2.imread('C:/Users/trainee.SEAHOST/Desktop/Sajitha p/Dataset(Pomegranate)/Bacterial Blight/bb_8.jpg')
#     #img=cv2.resize(img,(340,220))

#     #convert BGR to HSV
#     imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#     # create the Mask
#     mask=cv2.inRange(imgHSV,lowerBound,upperBound)
#     #morphology
#     maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
#     maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)

#     maskFinal=maskClose
#     conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
#     cv2.drawContours(img,conts,-1,(255,0,0),3)
#     for i in range(len(conts)):
#         x,y,w,h=cv2.boundingRect(conts[i])
#         cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)
#         cv2.putText(img, str(i+1),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255))
        
#     # cv2.imshow("maskClose",maskClose)
#     # cv2.imshow("maskOpen",maskOpen)
#     # cv2.imshow("mask",mask)
#     cv2.imshow("cam",img)

import cv2
import numpy as np
# from guidfilt import guided_filter
import matplotlib.pyplot as plt
from gflt import Guidedfilter
from GLRGB import *
from Glrm import GLRLM
import skimage.io
import skimage.color
from cv2.ximgproc import GuidedFilter
import skimage.filters
from PIL import Image
import time
from gflt import Guidedfilter
from sklearn.model_selection import train_test_split
from HONET import HOACAPS
from sklearn.metrics import accuracy_score  
import warnings
warnings.filterwarnings("ignore")
cl=[]
lowerBound=np.array([1,100,100])
upperBound=np.array([10,256,256])

#cam= cv2.VideoCapture(0)
kernelOpen=np.ones((10,10))
kernelClose=np.ones((30,30))

font = cv2.FONT_HERSHEY_SIMPLEX
path_to_images = "C:/Users/trainee.SEAHOST/Desktop/Sajitha p/Dataset(Pomegranate)/Scab/s_71.jpg"

image = cv2.imread(path_to_images)
# cv2.imshow('input',image)
#   #convert BGR to HSV
# imgHSV= cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
#   # create the Mask
# mask=cv2.inRange(imgHSV,lowerBound,upperBound)
#   #morphology
# maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
# maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
# maskFinal=maskClose
# conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(image,conts,-1,(255,0,0),3)
# for i in range(len(conts)):
#     x,y,w,h=cv2.boundingRect(conts[i])
#     cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255), 2)
#     cv2.putText(image, str(i+1),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255))
      
#   # cv2.imshow("maskClose",maskClose)
#   # cv2.imshow("maskOpen",maskOpen)
#   # cv2.imshow("mask",mask)
# cv2.imshow("cam",image)

I1=cv2.resize(image,(256,256))
cv2.imshow('Resized Image',I1)

'..........improved Guided Filter.............'
sigma_s = 1
sigma_r = 0.1*255
img_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
IF = Guidedfilter(I1, img_gray, sigma_s, sigma_r)
cv2.imshow('filtered',IF)

'...............Kmeans Clustering.....................'
cl_I = IF.reshape((-1,3))
cl_I  = np.float32(cl_I)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
attempts=10
ret,label,center=cv2.kmeans(cl_I ,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape((IF.shape))
cv2.imshow('kmeans',result_image)

grayImage = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)  
(thresh, bWI) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

cv2.imshow('binary',bWI)

seg=np.zeros((256,256))
for i in range(grayImage.shape[0]):
    for j in range(grayImage.shape[1]):
        if bWI[i,j]==255:
            seg[i,j]=0;
        else:
            seg[i,j]=255; 

cv2.imshow('seg',seg)

import skimage.measure

label_img = skimage.measure.label(seg)
regions = skimage.measure.regionprops(label_img)
Are=[]
Peri=[]
for props in regions:
    # print('Area:',props.area)
    area=props.area
    perimeter=props.perimeter
    # print('perimeter:',props.perimeter)
    Are.append(area)
    Peri.append(perimeter)
    
Area=Are[0]
Perimeter=Peri[0]
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
MER=[],
original = result_image.copy()
thresh = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
image = seg.copy()
# Find contours, obtain bounding box, extract and save ROI
ROI_number = 0
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255,0,0), 2)
    ROI = original[y:y+h, x:x+w]
    # cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
    ROI_number += 1
    # print("x,y,w,h:",x,y,w,h)
    X=[x,y,w,h]
Shapefet=[Area,Perimeter,x,y,w,h]
cv2.imshow('image', image)
# Shapefet=[Area,Perimeter,X]


# # rgb=[]
# # rgb_feat=(color_moments(result_image))

# # rest=result_image[:,:,-1]
# # properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
# # classes=['benign','malignant']
# # glcm_all_agls = []
# # glcm_all_agls=calc_glcm_all_agls(rest,0,props=properties)
# # app = GLRLM()
# # glrlm = app.get_features(result_image, 8)
# # glf1=glrlm.Features
# # glf2=glrlm.GLU
# # glf3=glrlm.LRE
# # glf4=glrlm.RLU
# # glf5=glrlm.RPC
# # glf6=glrlm.SRE
# # glf1.append(glf2)
# # glf1.append(glf3)
# # glf1.append(glf4)
# # glf1.append(glf5)
# # glf1.append(glf6)
# # glflfet=glf1.copy()
# # glflfeat=np.array(glflfet)
# # # print('GLFLM features',glflfeat)

# # shape_img_text_feat=np.asarray(glcm_all_agls + rgb_feat + Shapefet  + glflfet )
# # Feature=shape_img_text_feat.reshape(1,len(shape_img_text_feat))


# # Feat= np.load('features_400X.npy')
# # Label=np.load('labels_400X.npy')

# # X_train,X_test,Y_train,Y_test=train_test_split(Feat,Label,test_size=0.25,random_state=80)


# # hocaps=redecaps = HOACAPS(X_train,X_test,Y_train,Y_test)
# # pred=hocaps.HOA_CAPS()

# # print("acc",accuracy_score(Y_test,pred))

