# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:38:46 2022

@author: lfurtado
"""
import warnings
warnings.filterwarnings('ignore') 
import os
import numpy as np 
import pickle
from sklearn.metrics import classification_report
from simple_cnn import predict_CNN
from dialogue_rnn import predict_dialogue_RNN
from utils import build_batch, create_test_dataset

batch_size = 32

if os.path.exists('test_data.pkl'):
    dialogue_test_data = np.load(open('test_data.pkl', 'rb') ,allow_pickle=True)
else: 
    dialogue_test_data = create_test_dataset()
print(dialogue_test_data)
#data1 = build_batch(dialogue_train_data, 1, 8)
data2 = build_batch(dialogue_test_data, 2, batch_size)
data3 = build_batch(dialogue_test_data, 3, batch_size)
data4 = build_batch(dialogue_test_data, 4, batch_size)
data_test = data2 + data3 + data4

with open("emotion_label_decoder.pkl", "rb") as f:
    decoder_label = pickle.load(f)

y_pred, y_true, moyenne_accuracy = predict_CNN(data_test)
print("Test set CNN accuracy:")
print(moyenne_accuracy)
example = classification_report(y_true[0][0], y_pred[0][0])
print(example)

y_pred, y_true, moyenne_accuracy = predict_dialogue_RNN(data_test, batch_size)
print("Test set Dialogue RNN accuracy:")
print(moyenne_accuracy)

print(classification_report(y_true[0][0], y_pred[0][0]))
