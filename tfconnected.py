#!/usr/bin/env python
from keras.layers import Input, Dense,Conv2D,MaxPooling2D,Flatten,Reshape
from keras.models import Sequential
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

handle=open("mixed_images_tf_ravled.pickle","rb")
data=pickle.load(handle)
handle.close()

handle_=open("mixed_names_tf_ravled.pickle","rb")
lables=pickle.load(handle_)
handle_.close()
Data_Train,Data_test,Lables_train,Lables_test=train_test_split(data,lables,test_size=0.3,random_state=0)
epoch_=[200]
acc=[]
loss=[]
epoc_for_plot=[]
for x in epoch_:
 change=x;
 model = Sequential()
 model.add(Dense(units=60, activation='tanh',input_dim=40000))
 model.add(Dense(units=60, activation='relu'))
 model.add(Dense(units=3, activation='softmax'))
 model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
 graph=model.fit(Data_Train,Lables_train,validation_split=0.3,epochs=change)
 score=model.evaluate(Data_test,Lables_test)
 names=model.metrics_names
 
 acc.append(score[1])
 loss.append(score[0])
 epoc_for_plot.append(change)
 
 print(score,"score")
 print(names,"names")

 print(graph.history["val_loss"])
 plt.plot(graph.history["val_loss"])
 plt.ylabel('val_loss')
 plt.xlabel('epoch')
 #plt.legend(['Test_loss'],loc='upper left')
 #plt.show()
 plt.plot(graph.history["val_acc"])
 plt.ylabel("val_acc")
 plt.xlabel('epoch')
 plt.legend(['Validation_loss','Validation_acc'])
 plt.show()
 plt.plot(graph.history["loss"])
 plt.ylabel("Train_loss")
 plt.xlabel('epoch')
 #plt.legend(['Train_loss'],loc='upper left')
 #plt.show()
 plt.plot(graph.history["acc"])
 plt.ylabel("Train_acc")
 plt.xlabel('epoch')
 plt.legend(['Train_loss','Train_acc'])
 plt.show()
plt.plot(epoc_for_plot,acc)
plt.plot(epoc_for_plot,loss)
plt.ylabel("val_acc")
plt.xlabel('epoch')
plt.legend(['Test_acc','Test_loss'])
plt.show()

model.save("keras_tfmodel_200epoch")
