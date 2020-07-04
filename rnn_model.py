# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 18:02:21 2019

@author: Kyle
"""
import os

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import keras.backend as K
import tensorflow as tf
import keras
from keras.models import Model
from keras import regularizers

def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype= float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data

def load_data():    
    train_X = np.zeros([64, 16000])
    train_y = np.zeros([64])
    return train_X, train_y

def customLoss(yTrue, yPred):
    return K.sum(K.square(yTrue - yPred))


def lstm_model9_mfcc_fixed_scale(input, convLayer_num=4, filter_num=16, lstm_units=128, lstmLayer_num=2):
    input = tf.convert_to_tensor(input, dtype=tf.float32)
    
    # parameters of the network
    example_num = input.get_shape().as_list()[0]
    timeStep_num = 100
    subSequence_length = int(input.get_shape().as_list()[1]/timeStep_num)
    
    activationUnit = 'relu'
    
    # reshape into [ example_num * sequence, subsequence_length ]
    input = tf.reshape(input, [example_num * timeStep_num, 1, subSequence_length, 1])
    print(input.shape)
    
    # convLayer_num *( conv + maxpooling )
    for i in range(convLayer_num):
        input = keras.layers.convolutional.Conv2D(filter_num * (i+1), (1, 40), padding='same', activation= activationUnit)(input)
        print(input.shape)
        input = keras.layers.pooling.MaxPooling2D((1, 2), padding='same')(input)
        print(input.shape)
        print(i)
    
    # reshape for preparision of LSTM layers 
     # get the new sub-sequence length by multiplying the last two dimension ( old-sequence-length, convfeature_num )
    newSubSequence_length = np.multiply(*input.get_shape().as_list()[-2: ])
    input = tf.reshape(input, [example_num, timeStep_num, newSubSequence_length])
    
    # start the LSTM layers 
    for i in range(lstmLayer_num):
        input = tf.keras.layers.LSTM(int(lstm_units / (1)), activation= 'tanh', return_sequences=True, kernel_regularizer=regularizers.l2(0.001))(input)
    output = tf.reshape(input, [example_num * timeStep_num, int(lstm_units / 1)])
    output = keras.layers.core.Dense(max(32, int(lstm_units / 1)), activation='relu', kernel_regularizer=regularizers.l2(0.001))(output)
    output = keras.layers.core.Dense(max(32, int(lstm_units / 1 / 2)), activation='relu', kernel_regularizer=regularizers.l2(0.001))(output)

    output = keras.layers.core.Dense(5, activation='linear')(output)
    output = tf.reshape(output, [example_num, timeStep_num, 5])
    output = tf.transpose(output, perm=[0, 2, 1])
    output = tf.reshape(output, [example_num, timeStep_num * 5])
    return output


if __name__ == '__main__':
    train_X, train_y = load_data()
    test_X = train_X[0: 16, :]
    test_y = train_y[0: 16]
    model =  lstm_model9_mfcc_fixed_scale(train_X)