# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 00:54:40 2019

@author: Kyle
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from random import randint

batch_size = 160

def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.GFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

def run_graph(wav_data, input_layer_name='wav_data:0', output_layer_name='labels_softmax:0',
              num_top_predictions=3):
  """Runs the audio data through the graph and prints predictions."""
  
  with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions = sess.run(softmax_tensor, {'decode_wav:0': wav_data})

    return predictions
    
def inject_attack_5_fixed_scale(original_audio_csv, perturbation_info, batch_size=128, mode=0, scale=0.5):
    #fixed_scale = 32767 / 2
    fixed_scale = 32767 * scale
    if mode == 0:
        inject_success_count = 0
        perturbed_audio_csv = original_audio_csv.copy()
        perturbed_audio_csv = perturbed_audio_csv * 32767
        segment_num = 100
        file_num = int(original_audio_csv.shape[0] / batch_size) * batch_size
        inject_point_list = np.zeros([file_num, 5])
        
        for file_index in range(0, file_num):
            this_perturb_info = perturbation_info[file_index, :]
            this_perturb_info = np.reshape(this_perturb_info, [5, 100])
            for inject_index in range(0, 5):
                for i in range(0, segment_num - 1):

                    # if predicted time is earlier than the decision time, then immeditely inject the attack. We cannot inject attack to the past in the realtime setting.
                    # *100 or /100 is converting from the segment index (each segment is 0.01s) to time 
                    if this_perturb_info[inject_index, i] * 100 <= i + 1:
                        injection_point = max(i, this_perturb_info[inject_index, i] * 100) 
                        inject_point_list[file_index, inject_index] = injection_point / 100

                        position = min(int(injection_point * 160), 16000-160)
                        perturbed_audio_csv[file_index, position: position + 160] = perturbed_audio_csv[file_index, position: position + 160] + fixed_scale
                        inject_success_count += 1
                        break
        perturbed_audio_csv[perturbed_audio_csv>32767] = 32767
        perturbed_audio_csv = perturbed_audio_csv / 32768
    return perturbed_audio_csv, inject_point_list


def predict_classes_all(xs, data, target_class, minimize=True):
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    trail_num = xs.shape[0]
    #print(trail_num)
    if trail_num > batch_infer_size:
        print('error: ' + str(trail_num))
    data_perturbed = np.transpose(perturb_data(xs, data))
    input_data = np.zeros([16000, batch_infer_size])
    input_data[:, 0: trail_num] = data_perturbed / 32768
    
    predictions = run_graph(input_data)[0: trail_num]

    return predictions

def check_create_folder(folder_name):
    if os.path.exists(folder_name) == False:
        os.mkdir(folder_name)

if __name__ == '__main__':

    original_audio_csv = iter_loadtxt('original.csv')
    perturbation_info = iter_loadtxt('predicted_perturbation.csv')
    inject_attack_5_fixed_scale(original_audio_csv, perturbation_info)
