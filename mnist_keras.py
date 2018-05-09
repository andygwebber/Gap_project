# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 21:13:26 2018

@author: Andy Webber
"""

# Larger CNN for the MNIST Dataset

import numpy as np
import input_data
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.initializers import RandomNormal
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from copy import deepcopy
from imageset import ImageSet
K.set_image_dim_ordering('th')
import datetime
import pandas as pd

zfac = 0.30
epochs = 1
components = 100
iterations = 5
samples = 3

# define the larger model
def larger_model():
    # create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
     
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def run_example(restore = None, zerofac = 0.0):
    """ The run_example runs one example and returns the error. 
        The main program then does statistics on the errors"""
        

    """data = input_data.read_data_sets("MNIST_data/", one_hot=True)
    X_train_d, y_train_d = data.train.next_batch(50000)
    X_test_d, y_test_d = data.test.next_batch(11000)
    np.save("out_train_X", X_train_d)
    np.save("out_train_y", y_train_d)
    np.save("out_test_X", X_test_d)
    np.save("out_test_y", y_test_d) """

    X_train_d = np.load("out_train_X.npy")
    y_train_d = np.load("out_train_y.npy")
    X_test_d = np.load("out_test_X.npy")
    y_test_d = np.load("out_test_y.npy")

    N_clean = 1000
    N_images = X_train_d.shape[0]


    clean_images = ImageSet(np.copy(X_train_d[0:N_clean]), np.copy(y_train_d[0:N_clean]))
    clean_images.pca_calc(150)
    
#    train_X = np.copy(X_train_d[N_clean:N_images])
#    train_y = np.copy(y_train_d[N_clean:N_images])
#    dirty_train = ImageSet(train_X, train_y)

#    dirty_train = ImageSet(np.copy(X_train_d[N_clean:N_images]), np.copy(y_train_d[N_clean:N_images]))
#    dirty_test = ImageSet(np.copy(X_test_d), np.copy(y_test_d))
    
    dirty_train = ImageSet(X_train_d[N_clean:N_images], y_train_d[N_clean:N_images])
    dirty_test = ImageSet(X_test_d, y_test_d)

    
    if restore == None:
        pass
    else:
        dirty_train.dirty(zerofac)
        dirty_train.mean_calc()
        dirty_test.dirty(zerofac)
        dirty_test.mean_calc()
    if restore == 'pca_mean':
        dirty_train.recover_from_pca_mean(clean_images.pca)
        dirty_test.recover_from_pca_mean(clean_images.pca)
    elif restore == 'self_mean':
        dirty_train.recover_from_self_mean()
        dirty_test.recover_from_self_mean()
    elif restore == 'pca':
        dirty_train.recover_from_pca(clean_images.pca)
        dirty_test.recover_from_pca(clean_images.pca)
    elif restore == 'self_pca':
        dirty_train.recover_from_self_pca(components = components, iterations=iterations)
        dirty_test.recover_from_pca(dirty_train.pca)
#        dirty_test.recover_from_self_pca(components = 100, iterations=5)

    X_dirty_train = dirty_train.X
    X_dirty_test = dirty_test.X

    X_train = X_dirty_train.reshape(X_dirty_train.shape[0], 1, 28, 28).astype('float32')
    X_test = X_dirty_test.reshape(X_dirty_test.shape[0], 1, 28, 28).astype('float32')
# normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255

    y_train = dirty_train.y
    y_test = dirty_test.y

# build the model
    model = larger_model()
# Fit the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=100)
# Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Large CNN Error: %.2f%%" % (100-scores[1]*100))
    
    return (100-scores[1]*100)



start = datetime.datetime.now()

"""clean_errors = np.zeros(samples)
for i in range(samples):
    print("doing clean on sample ",i)
    error = run_example(restore = None, zerofac = zfac)
    clean_errors[i] = error

pca_mean_errors = np.zeros(samples)
for i in range(samples):
    print("doing pca_mean on sample ",i)
    error = run_example(restore = 'pca_mean', zerofac = zfac)
    pca_mean_errors[i] = error
    
self_mean_errors = np.zeros(samples)
for i in range(samples):
    print("doing self_mean on sample ",i)
    error = run_example(restore = 'self_mean', zerofac = zfac)
    self_mean_errors[i] = error
    
pca_errors = np.zeros(samples)
for i in range(samples):
    print("doing pca on sample ",i)
    error = run_example(restore = 'pca', zerofac = zfac)
    pca_errors[i] = error
    
self_pca_errors = np.zeros(samples)
for i in range(samples):
    print("doing self_pca on sample ",i)
    error = run_example(restore = 'self_pca', zerofac = zfac)
    self_pca_errors[i] = error"""
    
end = datetime.datetime.now()
print("The job took ", end-start)
   
""" Now output the result """
outfile = 'Result/result'
outfile += ('_zfac'+str(zfac))
outfile += ('_epochs'+str(epochs))
outfile += ('_comps'+str(components))
outfile += ('_iters'+str(iterations))

#np.savez(outfile,clean_errors,pca_mean_errors,self_mean_errors, pca_errors, self_pca_errors)

initial = [[None for _ in range(samples+3)] for _ in range(6)]
col_array = ["" for x in range(samples+2)]
for col in range(samples):
    col_array[col] = 'sample '+str(col)
col_array[samples] = 'mean'
col_array[samples+1] = 'standard deviation'
    
index_array = ["" for x in range(5)]

index_array[0] = "clean_errors"
#clean_errors = np.random.randint(10,size=samples)
for i in range(samples):
    initial[1][i+1] = clean_errors[i]
initial[1][samples+1] = np.mean(clean_errors)
initial[1][samples+2] = np.std(clean_errors)
    
index_array[1] = "pca_mean_errors"
#pca_mean_errors = np.random.randint(10,size=samples)
for i in range(samples):
    initial[2][i+1] = pca_mean_errors[i]
initial[2][samples+1] = np.mean(pca_mean_errors)
initial[2][samples+2] = np.std(pca_mean_errors)
    
index_array[2] = "self_mean_errors"
#self_mean_errors = np.random.randint(10,size=samples)
for i in range(samples):
    initial[3][i+1] = self_mean_errors[i]
initial[3][samples+1] = np.mean(self_mean_errors)
initial[3][samples+2] = np.std(self_mean_errors)

index_array[3] = "pca_errors"
#pca_errors = np.random.randint(10,size=samples)
for i in range(samples):
    initial[4][i+1] = pca_errors[i]
initial[4][samples+1] = np.mean(pca_errors)
initial[4][samples+2] = np.std(pca_errors)
    
index_array[4] = "self_pca_errors"
#self_pca_errors = np.random.randint(10,size=samples)
for i in range(samples):
    initial[5][i+1] = self_pca_errors[i]
initial[5][samples+1] = np.mean(self_pca_errors)
initial[5][samples+2] = np.std(self_pca_errors)

data3_np = np.array(initial)

data3_df = pd.DataFrame(data=data3_np[1:,1:],
                        index=np.array(index_array),
                        columns=np.array(col_array))
outfile += '.csv'

data3_df.to_csv(outfile, sep=',')
  
