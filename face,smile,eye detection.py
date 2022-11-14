#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2  
from matplotlib import pyplot as plt  
import numpy as np  
 
  
  
 
  
 
font=cv2.FONT_HERSHEY_PLAIN  
  
faceClassifier=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')  
eyeClassifier=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')  
smileClassifier=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')  
  
img=cv2.imread('C:/Users/emy/Desktop/viola.jpg') 
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
plt.figure(figsize=(12,12))  
faces=faceClassifier.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=2,minSize=(70,70))  
  
for x,y,w,h in faces: 
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),5)  
    cv2.putText(img,"Peter",(x,y+h+40),font,3,(255,0,0),2,cv2.LINE_AA)  
  
    face=gray[y:y+h,x:x+w]  
    eyes=eyeClassifier.detectMultiScale(face,scaleFactor=1.05,minNeighbors=5,minSize=(20,20))  
for x2,y2,w2,h2 in eyes: 
    cv2.rectangle(img,(x+x2,y+y2),(x+x2+w2,y+y2+h2),(0,255,0),2)  
    cv2.putText(img,"eye",(x+x2,y+y2+h2+12),font,1,(255,0,0),1,cv2.LINE_AA)  
          
    smiles=smileClassifier.detectMultiScale(face,scaleFactor=1.1,minNeighbors=6,minSize=(90,90)) 
for x2,y2,w2,h2 in smiles:  
    cv2.rectangle(img,(x+x2,y+y2),(x+x2+w2,y+y2+h2),(0,255,0),2)  
    cv2.putText(img,"smile",(x+x2,y+y2+h2+12),font,1,(255,0,0),1,cv2.LINE_AA)  
  
  
plt.imshow(img[:,:,::-1])
plt.show()


# In[ ]:




