import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm


X_train = []
Y_train = []

X_test = []
Y_test = []



DATADIR = "~/Research/OneraDataset"
path1 = os.path.join(DATADIR,"train1")
path2 = os.path.join(DATADIR,"train2")  
'''
os.mkdir("OneraData/train")
os.mkdir("OneraData/val")
os.mkdir("OneraData/test")
os.mkdir("OneraData/train/class_a")
os.mkdir("OneraData/train/class_b")
os.mkdir("OneraData/test/class_a")
os.mkdir("OneraData/test/class_b")
os.mkdir("OneraData/val/class_a")
os.mkdir("OneraData/val/class_b")
'''
for i in range(len(os.listdir(path1))):
  img1 = os.listdir(path1)[i]
  img2 = os.listdir(path2)[i]   
  img_array1 = cv2.imread(os.path.join(path1,img1) ,cv2.IMREAD_GRAYSCALE)  
  img_array2 = cv2.imread(os.path.join(path2,img2) ,cv2.IMREAD_GRAYSCALE)
  img_array1 = cv2.resize(img_array1,(500,500))
  img_array2 = cv2.resize(img_array2,(500,500))   
  #stackedImg = tf.stack([img_array1, img_array2])
  stackedImg = np.concatenate((img_array1, img_array2), axis=1)
  X_train.append(stackedImg)
  im = Image.fromarray(stakedImg)
  im.save("OneraData/train/class_a/class_a_" + str(i) + ".png")


path3 = os.path.join(DATADIR,"test1")
path4 = os.path.join(DATADIR,"test2")  

for i in range(len(os.listdir(path3))):
  img3 = os.listdir(path3)[i]
  img4 = os.listdir(path4)[i]    
  img_array3 = cv2.imread(os.path.join(path3,img3) ,cv2.IMREAD_GRAYSCALE)  
  img_array4 = cv2.imread(os.path.join(path4,img4) ,cv2.IMREAD_GRAYSCALE)
  img_array3 = cv2.resize(img_array3,(500,500))
  img_array4 = cv2.resize(img_array4,(500,500))
  #stackedImg = tf.stack([img_array3,img_array4])    
  stackedImg = np.concatenate((img_array3, img_array4), axis=1)
  X_test.append(stackedImg)
  im = Image.fromarray(stackedImg)
  im.save("OneraData/test/class_a/class_a_" + str(i) + ".png")


path3 = os.path.join(DATADIR,"val1")
path4 = os.path.join(DATADIR,"val2")  

for i in range(len(os.listdir(path3))):
  img3 = os.listdir(path3)[i]
  img4 = os.listdir(path4)[i]    
  img_array3 = cv2.imread(os.path.join(path3,img3) ,cv2.IMREAD_GRAYSCALE)  
  img_array4 = cv2.imread(os.path.join(path4,img4) ,cv2.IMREAD_GRAYSCALE)
  img_array3 = cv2.resize(img_array3,(500,500))
  img_array4 = cv2.resize(img_array4,(500,500))
  #stackedImg = tf.stack([img_array3,img_array4])    
  stackedImg = np.concatenate((img_array3, img_array4), axis=1)
  X_test.append(stackedImg)
  im = Image.fromarray(stackedImg)
  im.save("OneraData/val/class_a/class_a_" + str(i) + ".png")

path5 = os.path.join(DATADIR,"target_val")

for i in range(len(os.listdir(path5))):
  img5 = os.listdir(path5)[i]
  img_array5 = cv2.imread(os.path.join(path5,img5) ,cv2.IMREAD_GRAYSCALE)
  img_array5 = cv2.resize(img_array5,(500,500))     
  img_array = np.zeros_like(img_array5)
  stackedImg = np.concatenate((img_array, img_array5), axis=1)
  Y_train.append(stackedImg)
  im = Image.fromarray(stackedImg)
  im.save("OneraData/val/class_b/class_b_" + str(i) + ".png")

path5 = os.path.join(DATADIR,"target_train")

for i in range(len(os.listdir(path5))):
  img5 = os.listdir(path5)[i]
  img_array5 = cv2.imread(os.path.join(path5,img5) ,cv2.IMREAD_GRAYSCALE)
  img_array5 = cv2.resize(img_array5,(500,500))     
  img_array = np.zeros_like(img_array5)
  stackedImg = np.concatenate((img_array, img_array5), axis=1)
  Y_train.append(stackedImg)
  im = Image.fromarray(stackedImg)
  im.save("OneraData/train/class_b/class_b_" + str(i) + ".png") 
  

path6 = os.path.join(DATADIR,"target_test")

for i in range(len(os.listdir(path6))):
  img6 = os.listdir(path6)[i]
  img_array6 = cv2.imread(os.path.join(path6,img6) ,cv2.IMREAD_GRAYSCALE)
  img_array6 = cv2.resize(img_array6,(500,500))   
  img_array = np.zeros_like(img_array6)
  stackedImg = np.concatenate((img_array, img_array6), axis=1)
  Y_test.append(stackedImg)
  im = Image.fromarray(stackedImg)
  im.save("OneraData/train/class_b/class_b_" + str(i) + ".png")

X_train = np.asarray(X_train)
Y_train = np.asarray(X_train)

X_train = X_train/225
Y_train = Y_train/225

X_test = np.asarray(X_test)
Y_test = np.asarray(X_test)

X_test = X_test/225
Y_test = Y_test/225
