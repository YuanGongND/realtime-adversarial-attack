# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 19:27:54 2019

@author: Yuan Gong

# implementation of one-pixel-attack on voice command system


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from sys import argv
_, newFolderName, gpuI = argv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuI)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import math 

from scipy.io import wavfile
import tensorflow as tf
import numpy as np
import shutil

# single thread, very slow
#from scipy.optimize import differential_evolution

from differential_evolution import differential_evolution 

times = 0
# batch_size refers the pertubation segment length, here we use 0.01 seconds = 160 @ 16kHz, see the "batch processing" section of the paper
batch_size = 160

# batch_infer_size refers to the number of expert demonstrations generated in a batch
batch_infer_size = 50

labels = 'conv_labels.txt'
labels_list = [line.rstrip() for line in tf.gfile.GFile(labels)]

#%% creat folder to save model 
newFolderName = 'perturbed_data/' + newFolderName
while os.path.exists(newFolderName) == True:
    newFolderName = newFolderName + '_1'
    #newFolderName = 'tmp_' + str( time.strftime('%Y_%m_%d_%H_%M',time.localtime(time.time())) )
os.mkdir(newFolderName)
shutil.copy('formal_generate.py', newFolderName)


#%%
def load_data(filename):
    fs, data = wavfile.read(filename)
    return data


def perturb_data(xs, input_data):

    perturbation_scale = 0.25
    perturb_amp = int(32767 * perturbation_scale)
    
    data = input_data.copy()
    original_data = input_data.copy()

    # if perturb multiple points, in this work, we perturb 5 points
    if xs.ndim > 1:
        trail_num = xs.shape[0]
        perturb_num = int(xs.shape[1])
        data = np.tile(data, [trail_num, 1])
        for trail_index in range(0, trail_num):
            for perturb_index in range(0, perturb_num):
                position = min(int(xs[trail_index, perturb_index] * batch_size), 16000 - batch_size)
                perturbed_part = data[trail_index, position: position + batch_size] + perturb_amp
                
                # clip the data over 32767 
                perturbed_part[perturbed_part > 32767] = 32767
                perturbed_part = perturbed_part.astype('int16')
                data[trail_index, position: position + batch_size] = perturbed_part
    
    # if only perturb 1 points, i.e., xs, the evolution optimizer output, is one dimensional vector
    else:
        perturb_num = len(xs)
        for perturb_index in range(0, perturb_num):
            position = min(int(xs[perturb_index] * batch_size), 16000 - batch_size)
            perturbed_part = data[position: position + batch_size] + perturb_amp
            perturbed_part[perturbed_part >= 32767] = 32767
            perturbed_part = perturbed_part.astype('int16')
            data[position: position + batch_size] = perturbed_part
    
    # return the perturbed data
    return data

# get the prediction confidence of the target model of a GIVEN class
def predict_classes(xs, data, target_class, minimize=True):
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    trail_num = xs.shape[0]
    if trail_num > batch_infer_size:
        print('error: ' + str(trail_num))
    data_perturbed = np.transpose(perturb_data(xs, data))
    input_data = np.zeros([16000, batch_infer_size])
    input_data[:, 0: trail_num] = data_perturbed / 32768
    
    predictions = run_graph(input_data)[0: trail_num, target_class]

    return predictions 

# get the prediction confidence of the target model of all classes (i.e., predictive distribution)
def predict_classes_all(xs, data, target_class, minimize=True):
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    trail_num = xs.shape[0]
    if trail_num > batch_infer_size:
        print('error: ' + str(trail_num))
    data_perturbed = np.transpose(perturb_data(xs, data))
    input_data = np.zeros([16000, batch_infer_size])
    input_data[:, 0: trail_num] = data_perturbed / 32768
    
    predictions = run_graph(input_data)[0: trail_num]

    return predictions


# this is mainly from the one-pixel attack paper
def attack_fix_scale(input_data, ground_truth_label, target=None, verbose=False, pixel_count=1,
           maxiter=5, popsize=10, seed=17):
    # Change the target class based on whether this is a targeted attack or not
    data = input_data.copy()
    targeted_attack = target is not None
    target_class = target if targeted_attack else ground_truth_label
    
    # Define bounds for a flat vector of x,y,r,g,b values
    # For more pixels, repeat this layout
    bounds = [(0, int(16000 / batch_size))] * pixel_count
    
    # Population multiplier, in terms of the size of the perturbation vector x
    popmul = max(1, popsize // len(bounds))
    
    # Format the predict/callback functions for the differential evolution algorithm
    predict_fn = lambda xs: predict_classes(xs, data, target_class)

    
    # Call Scipy's Implementation of Differential Evolution
    attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, recombination=1, 
                                           atol=-1, popsize=popsize, polish=False, seed=seed)
    
    return attack_result.x

def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.GFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def run_graph(wav_data, input_layer_name='wav_data:0', output_layer_name='labels_softmax:0',
              num_top_predictions=3):
  """Runs the audio data through the graph and prints predictions."""
  
  global times
  times += 1

  with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions = sess.run(softmax_tensor, {'decode_wav:0': wav_data})

#    # Sort to show labels in order of confidence
#    top_k = predictions.argsort()[-num_top_predictions:][::-1]
#    for node_id in top_k:
#      human_string = labels_list[node_id]
#      score = predictions[node_id]
#      #print('%s (score = %.5f)' % (human_string, score))
#    #print('-------------------------------------------')

    return predictions

def write_wavfile(filename, data):
    wavfile.write(filename, 16000, data)


# maxiter and popsize are the parameters for the evolution algorithm
def generate_perturbation_fix_scale(wav_filename, ground_truth_class, seed=17, pixel_count=1, maxiter=5, popsize=10, testname='formal_test'):
    
    global times
    times = 0
    start = time.time()

    data = load_data(wav_filename)
    if len(data) != 16000:
        print('adjusted')
        data_pad = np.zeros(16000)
        data_pad[0: len(data)] = data
        data = data_pad
    
    result = np.zeros(6)

    # this is the key function
    attack_result = attack_fix_scale(data, ground_truth_class, pixel_count=pixel_count, maxiter=maxiter, popsize=popsize)
    
    wav_filename = wav_filename[0:len(wav_filename)-4]
    wav_filename = wav_filename.split('/')[-1]
    np.savetxt(testname + '/' + wav_filename+ '.csv', attack_result, delimiter = ',')
    write_wavfile(testname + '/' + wav_filename+'_perturbed.wav', perturb_data(attack_result, data))
    
    end = time.time()
    print('time per round: %.2f seconds' % ((end-start)))
    print('time per round: %.3f seconds' % ((end-start)/times/(pixel_count * maxiter * popsize)))
    
    xs = np.array([[-1] * pixel_count, attack_result])
    predicted_probs = predict_classes_all(xs, data, ground_truth_class)
    cdiff = predicted_probs[0, ground_truth_class] - predicted_probs[1, ground_truth_class]
    prior_class = np.argmax(predicted_probs[0])
    poster_class = np.argmax(predicted_probs[1])
    if prior_class != poster_class:
        success = 1
    else:
        success = 0
    print("prior class: %d, poster class: %d, prior: %.2f, posterier: %2f, cdiff: %2f, success: %d" % (prior_class, poster_class, predicted_probs[0, ground_truth_class], predicted_probs[1, ground_truth_class], cdiff, success))
    data = perturb_data(attack_result, data)
    result[0: 6] = [prior_class, poster_class, predicted_probs[0, ground_truth_class], predicted_probs[1, ground_truth_class], cdiff, success]
        
    return result


def check_create_folder(folder_name):
    if os.path.exists(folder_name) == False:
        os.mkdir(folder_name)
    

if __name__ == '__main__':
    
    # perturb how many segments of the audio
    pixel_count = 5   
    success_count = 0
    total_file_count = 0
    
    # load the target speech recognition model
    graph = 'speech_model_train/trained_models/formal_model_' + str(batch_infer_size) + '.pb'
    load_graph(graph)

    # load the data used for generating expert demonstrations
    base_path = 'dataset/attack_train/'
    rootdir_list = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    result_all = np.zeros([1,6])
    
    for keyword_index in range(0, 10):
        file_list = os.listdir(base_path + rootdir_list[keyword_index])
        result = np.zeros([len(file_list), 6])
        for i in range(0, len(file_list)):
            path = os.path.join(base_path + rootdir_list[keyword_index], file_list[i])
            
            print(newFolderName + '/' + rootdir_list[keyword_index])
            check_create_folder(newFolderName + '/' + rootdir_list[keyword_index] + '/')
            if os.path.isfile(path):
                print('---------------------------------------')
                print('File: ' + str(i) + '/' + str(len(file_list)) +' : ' + path)
                
                # generate_perturbation_fix_scale is the key function, keyword_index+2 because the first two keyword is silence and unknown.
                result[i, :] = generate_perturbation_fix_scale(path, keyword_index + 2, pixel_count=pixel_count, testname = newFolderName)
                success_count += result[i, 5]
                total_file_count += 1
                print('success %d/ total %d = %.3f' % (success_count, total_file_count, success_count / total_file_count))
                np.savetxt(newFolderName + '/' + rootdir_list[keyword_index] + '/' + 'result.csv', result, delimiter=',')
        result_all = np.concatenate((result_all, result))
        np.savetxt(newFolderName + '/result_all.csv', result_all, delimiter=',')
    tf.reset_default_graph()
   
