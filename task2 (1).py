#!/usr/bin/env python
# coding: utf-8

# In[6]:


import streamlit as st
import cv2
import numpy as np

#set title for the application
st.title("My First Streamlit Application")

st.write("Please Upload Your Image")



# In[7]:


config_file="C:/Users/ASUS/Downloads/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model="C:/Users/ASUS/Downloads/frozen_inference_graph.pb"


# In[14]:
model=cv2.dnn_DetectionModel(frozen_model,config_file)
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5)) 
model.setInputSwapRB(True)
class_labels=[]
file_name="C:/Users/ASUS/Downloads/labels.txt"
with open (file_name,'rt')as fpt:
    class_labels=fpt.read().rstrip("\n").split("\n")



# In[9]:


#display a file uploader widget
uploaded_image = st.file_uploader("choose an Image..", type = ['jpg', 'jpeg', 'png'])



#In[13]:


#if an image is uploaded, display it
if uploaded_image is not None:
    image = np.array(bytearray(uploaded_image.read()), dtype = np.uint8)
    
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    #display image
    st.image(uploaded_image, caption = 'BGR Image.', channels = 'BGR')


    
    ClassIndex,confidece,bbox=model.detect(img,confThreshold=0.5)
    font_scale=3
    font=cv2.FONT_HERSHEY_PLAIN
    org=(443,320)
    for ClassInd ,conf, boxes in zip(ClassIndex.flatten(),confidece.flatten(),bbox):
        cv2.rectangle(img,boxes,(225,0,0),2)
        cv2.putText(img,class_labels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,225,0),thickness=3)
        imd=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    st.image(imd, caption = 'detect objects', channels = 'RGB')
    
# In[ ]:



# In[ ]:



    



# %%
