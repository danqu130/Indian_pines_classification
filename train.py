
# coding: utf-8

# In[1]:


import numpy as np
import scipy
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.utils import np_utils
#from sklearn.cross_validation import StratifiedKFold


# In[2]:


PATH = os.getcwd()
print (PATH)


# In[3]:


# Global Variables
windowSize = 5
numPCAcomponents = 30
testRatio = 0.25


# # Load Training Dataset

# In[4]:


X_train = np.load("./predata/XtrainWindowSize" 
                  + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio)  + ".npy")

y_train = np.load("./predata/ytrainWindowSize" 
                  + str(windowSize) + "PCA" + str(numPCAcomponents) + "testRatio" + str(testRatio) + ".npy")


# In[5]:


# Reshape into (numberofsumples, channels, height, width)
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[3], X_train.shape[1], X_train.shape[2]))


# In[6]:


# convert class labels to on-hot encoding
y_train = np_utils.to_categorical(y_train)


# In[7]:


# Define the input shape 
input_shape= X_train[0].shape
print(input_shape)


# In[8]:


# number of filters
C1 = 3*numPCAcomponents


# In[9]:


# Define the model
model = Sequential()

model.add(Conv2D(C1, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(3*C1, (3, 3), activation='relu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(6*numPCAcomponents, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='softmax'))


# In[10]:


sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# In[11]:


model.fit(X_train, y_train, batch_size=32, epochs=30)


# In[12]:


import h5py
from keras.models import load_model


# In[13]:


model.save('./model/HSI_model_epochs30.h5')

