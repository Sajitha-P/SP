


import numpy as np
import os
import cv2
from keras.utils import array_to_img
from keras.utils import img_to_array
from keras.utils import load_img
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from numpy import expand_dims
path = "dataset/Scab"
files = os.listdir(path)
files = [x for x in files if x !='Thumbs.db']
#################################################################
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(featurewise_center=True,rotation_range=20,width_shift_range=0.1,brightness_range = (0.5, 1.1),
                             height_shift_range=0.1,horizontal_flip=True,zoom_range = 0.1)
for file in files:
    root = path+"/"+file
    """ read the image """
    image = cv2.imread(root)
    print(root)
    data = img_to_array(image)
    samples = expand_dims(data, 0)
    i=0
    lab_name = file.split(".")[0].split("-")[-1]
    for batch in datagen.flow(samples, batch_size = 1):
        bt = batch[0].astype(np.uint8)
        cv2.imshow("dfdfs",bt)
        
        path_save = "augment_images//"+lab_name+"-"+str(i)+".png"
        cv2.imwrite(path_save,bt)
        i+= 1
        if i > 15:break


