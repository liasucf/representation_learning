# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:38:46 2022

@author: lfurtado
"""
import warnings
warnings.filterwarnings('ignore') 
import os
import numpy as np 
from sklearn.metrics import classification_report
from simple_cnn import predict_CNN
from dialogue_rnn import predict_dialogue_RNN
from utils import build_batch, create_test_dataset
import time


"""Code to compare the DialogueRNN and CNN networks in the emotion classification
problem for the Friends Dataset

Test the networks in the test data and analyse accuracy and other metrics
"""

batch_size = 32

#loade test data
if os.path.exists('test_data.pkl'):
    test_data = np.load(open('test_data.pkl', 'rb') ,allow_pickle=True)
else: 
    dialogue_test_data = create_test_dataset()

#create batchs from the test data with conversations of 2,3 and 4 people
data2 = build_batch(dialogue_test_data, 2, batch_size)
data3 = build_batch(dialogue_test_data, 3, batch_size)
data4 = build_batch(dialogue_test_data, 4, batch_size)
data_test = data2 + data3 + data4

# ---------- predict CNN
start_time = time.time()
y_pred, y_true, moyenne_accuracy = predict_CNN(data_test)

final_time = time.time() - start_time
print("Test set CNN accuracy:")
print(moyenne_accuracy)
print("Time")
print(final_time)

y_pred_conversation_1 = [item for sublist in y_pred[0] for item in sublist]
y_true_conversation_1 = [item for sublist in y_true[0] for item in sublist]


print(classification_report(y_true_conversation_1, y_pred_conversation_1))

# ---------- predict Dialogue RNN

start_time = time.time()
y_pred, y_true, moyenne_accuracy = predict_dialogue_RNN(data_test, batch_size)
print("Test set Dialogue RNN accuracy:")
print(moyenne_accuracy)
final_time = time.time() - start_time
print("Time")
print(final_time)

y_pred_conversation_1 = [item for sublist in y_pred[0] for item in sublist]
y_true_conversation_1 = [item for sublist in y_true[0] for item in sublist]


print(classification_report(y_true_conversation_1, y_pred_conversation_1))

#if necessary to decode the labels to emotion
#with open("emotion_label_decoder.pkl", "rb") as f:
    #decoder_label = pickle.load(f)