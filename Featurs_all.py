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
from rgb import color_moments
import time
import os
from gflt import Guidedfilter
import warnings
warnings.filterwarnings("ignore")
cl=[]
folder_dir = "Dataset(Pomegranate)"  
Features_all=np.zeros((1,49))
Label=[]
Label.append(1)
for fold in os.listdir(folder_dir):
    root=folder_dir+"/"+fold
    for images in os.listdir(root):
        path_img=root+"/"+images
        print(path_img)
        image = cv2.imread(path_img)
        if 'Normal'in path_img:
            Label.append(0)
        elif  'Bacterial Blight' in path_img:
            Label.append(1)
        elif  'Heart rot' in path_img:
            Label.append(2)
        else:
            Label.append(3)
        I1=cv2.resize(image,(256,256))        
        '..........improved Guided Filter.............'
        sigma_s = 1
        sigma_r = 0.1*255
        img_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
        IF = Guidedfilter(I1, img_gray, sigma_s, sigma_r)
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
        grayImage = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)  
        (thresh, bWI) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
        seg=np.zeros((256,256))
        for i in range(grayImage.shape[0]):
            for j in range(grayImage.shape[1]):
                if bWI[i,j]==255:
                    seg[i,j]=0;
                else:
                    seg[i,j]=255;         
        import skimage.measure
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
        MER=[],
        original = result_image.copy()
        thresh = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        image = seg.copy()
        ROI_number = 0
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            ROI = original[y:y+h, x:x+w]
            ROI_number += 1
            X=[x,y,w,h]
        Shapefet=([Area,Perimeter,x,y,w,h])        
        
        rgb=[]
        rgb_feat=(color_moments(result_image))
        
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
        Features_all=np.concatenate((Features_all,Feature),axis=0)
np.save("Features_all.npy",Features_all)      
np.save("labels_all.npy",Label)