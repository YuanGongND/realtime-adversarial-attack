# -*- coding: utf-8 -*-

import os
from sys import argv
_, newFolderName, gpuI = argv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuI)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from tensorflow.python.platform import tf_logging as logging
from keras import backend as K
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.models import load_model
import time
import shutil
import rnn_model
from random import randint
import inject_attack_callable


#%% creat folder to save model 
newFolderName = r'../experiment/' + newFolderName
while os.path.exists(newFolderName) == True:
    newFolderName = newFolderName + '_1'
    #newFolderName = 'tmp_' + str( time.strftime('%Y_%m_%d_%H_%M',time.localtime(time.time())) )
os.mkdir(newFolderName)
shutil.copy('rnn_model.py', newFolderName)
shutil.copy(os.path.basename(__file__), newFolderName)


#%% sub-routines
# fast read data function( very useful in large data )
def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype=float):
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


def process_missing_value(data_X):
    where_are_nan = np.isnan(data_X) 
    where_are_inf = np.isinf(data_X) 
    data_X[where_are_nan] = 0 
    data_X[where_are_inf] = 0 
    
    std = StandardScaler()
    data_X = std.fit_transform(data_X)

def load_data_complete(ori_train_filename, per_train_filename, test1_filename, test2_filename):
    train_X = iter_loadtxt(ori_train_filename)
    train_y = iter_loadtxt(per_train_filename)

    test1_X = iter_loadtxt(test1_filename)
    test2_X = iter_loadtxt(test2_filename)
    
    train_X = train_X / np.abs(train_X).max(axis=1)[:, None]
    test1_X = test1_X / np.abs(test1_X).max(axis=1)[:, None]
    test2_X = test2_X / np.abs(test2_X).max(axis=1)[:, None]
    
    # print shapes
    print(train_X.shape)
    print(train_y.shape)
    print(test1_X.shape)
    print(test2_X.shape)
    
    return train_X, train_y, test1_X, test2_X

def load_data(ori_train_filename, per_train_filename, ori_test_filename):
    train_X = iter_loadtxt(ori_train_filename)
    train_y = iter_loadtxt(per_train_filename)

    test_X = iter_loadtxt(ori_test_filename)
    #test_y = iter_loadtxt(per_test_filename)
    
    train_X = train_X / np.abs(train_X).max(axis=1)[:, None]
    test_X = test_X / np.abs(test_X).max(axis=1)[:, None]
    
    print(train_X.shape)
    print(train_y.shape)
    print(test_X.shape)
    return train_X, train_y, test_X

def train(train_X, train_y, test_X, batch_size=64, lr=1e-4, iteration_num=100, subfolder_name='', convLayer_num=2, filter_num=32, lstm_units=32, lstmLayer_num=1):
    
    train_datasize = train_X.shape[0]
    test_datasize = test_X.shape[0]
    result = np.zeros([2, iteration_num]) 
    if subfolder_name != '':
        os.mkdir(newFolderName+subfolder_name)
        
    with tf.Session() as sess:        
        global_step = tf.Variable(0)  
        learning_rate = tf.train.exponential_decay(lr, global_step, int(2 * iteration_num*(train_datasize/batch_size)), 1, staircase=False)  
        tf.set_random_seed(17)
        input_x = tf.placeholder(tf.float32, shape=(batch_size, 16000), name = 'inputx')
        input_y = tf.placeholder(tf.float32, shape=(batch_size, 500), name = 'inputy')
        
        # feed the input to the model
        prediction = rnn_model.lstm_model9_mfcc_fixed_scale(input_x, lstm_units=lstm_units, lstmLayer_num=lstmLayer_num)
        
        time_scale = np.array(list(range(0,100)) * 5) / 100
        time_scale = tf.cast(tf.constant(time_scale),'float32')
        
        loss =  tf.sqrt(tf.losses.mean_squared_error(labels=input_y, predictions=prediction))
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        saver = tf.train.Saver()        
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        # number of iterations
        for iteration in range( 0, iteration_num ):
            
            print('---------------------')
            print('iteration: ' + str(iteration) + '  learning_rate: ' + str(learning_rate.eval()))
            
            # shuffle the training set
            np.random.seed(seed=7)
            np.random.shuffle(train_X)
            np.random.seed(seed=7)
            np.random.shuffle(train_y)
#            
            start_time = time.time()
            
            # train stage
            loss_list = []
            for i in range(0, 1 *int(train_datasize / batch_size)):
                start = (i * batch_size) % train_datasize
                end = min(start + batch_size, train_datasize)
                
                inputTrainFeature = train_X[start:end, :]
                inputTrainLabel = train_y[start:end, :]
                
                augment_train_feature = inputTrainFeature
                augment_train_label = inputTrainLabel

                _, lossShow, prediction_show = sess.run([train_step, loss, prediction], feed_dict = {input_x:augment_train_feature, input_y:augment_train_label})
                
                loss_list.append(lossShow)

            # save the last batch
            if iteration % 10 == 0 or iteration < 200: 
                np.savetxt(newFolderName + subfolder_name + '/train_predictions_' + str(iteration) + '.csv', prediction_show, delimiter = ',' )
                np.savetxt(newFolderName + subfolder_name + '/train_label_' + str(iteration) + '.csv', augment_train_label, delimiter = ',' )
                np.savetxt(newFolderName + subfolder_name + '/train_origin_' + str(iteration) + '.csv', augment_train_feature, delimiter = ',' )
                
            # test stage
            if iteration % 10 == 0 or iteration < 200:
                test_loss_list = []
                test_predictions_all = np.zeros([int(test_datasize / batch_size) * batch_size, train_y.shape[1]])
                for i in range(0, int(test_datasize / batch_size)):
                    start = (i * batch_size)
                    end = start + batch_size
                    
                    inputTestFeature = test_X[start:end, :]
                    
                    prediction_show, lossShow = sess.run([prediction, loss], feed_dict = {input_x:inputTestFeature})
                    test_predictions_all[start:end] = prediction_show
                        
                    test_loss_list.append(lossShow)
                np.savetxt(newFolderName + subfolder_name + '/test_predictions_all_' + str(iteration) + '.csv', test_predictions_all, delimiter = ',')

            result[0, iteration] = np.mean(loss_list)
            result[1, iteration] = np.mean(test_loss_list)

            print('train loss = ' + str(np.mean(loss_list)) + ' , test loss = ' + str(np.mean(test_loss_list)))
            np.savetxt(newFolderName + subfolder_name + '/result.csv', result, delimiter = ',' )
            # save model every 10 epoches
            if iteration%10 == 0:
                save_path = saver.save(sess, newFolderName + subfolder_name + '/model_' + str(iteration) + '_.ckpt')
                print("Model saved in file: %s" % save_path)
            end_time = time.time()
            print('time:' + str(end_time-start_time) + ' seconds')
            print('time:' + str((end_time-start_time)/60) + ' minutes')
            
    return result

if __name__ == '__main__':
    
    ori_test_filename = r'..\ready_to_train_files\\attack_test1_pixel_5_fixed_scale\original.csv'
    ori_train_filename = r'..\ready_to_train_files\\attack_train_pixel_5_fixed_0.25_scale\original.csv'
    per_train_filename = r'..\ready_to_train_files\\attack_train_pixel_5_fixed_0.25_scale\perturbation.csv'

    train_X, train_y, test_X = load_data(ori_train_filename, per_train_filename, ori_test_filename)
    iteration_num = 150
    result_all = train(train_X, train_y, test_X, lr=1e-3, iteration_num=iteration_num, lstm_units=256, lstmLayer_num=2, subfolder_name='')
    np.savetxt(newFolderName + '/result_all.csv', result_all, delimiter=',')
    tf.reset_default_graph()
