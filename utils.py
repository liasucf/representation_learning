# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 09:26:48 2022

@author: lfurtado
"""
import os
import json
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.preprocessing.text import Tokenizer
import contractions
from sklearn.preprocessing import LabelBinarizer

"Convert a numpy array into a tensor"
def convert_tensor(tensor):
    np_tensor = tensor.numpy()
    tensor = tf.convert_to_tensor(np_tensor, dtype="float32")

    return tensor

"deal with contracted texts"
def expand_text(text):
    expanded_words = []
    text = text.encode('utf-8').decode('cp1252').replace("Â’", "'")
    #text = text.replace("’", "'")
    for word in text.split():
      # using contractions.fix to expand the shotened words
      expanded_words.append(contractions.fix(word))   

    expanded_text = ' '.join(expanded_words)
    return expanded_text

"clean dataset"
def preprocess_text(x):
    for punct in '"!&?.,}-/<>#$%\()*+:;=?@[\\]^_`|\~':
        x = x.replace(punct, ' ')
    x = ' '.join(x.split())
    x = x.lower()
    
    return x

"transform the json into a dataframe"
def create_sentences(filename, split):
    sentences, emotion_labels, speakers, conv_id, = [], [], [], []
    
    with open(filename, 'r', encoding='latin1') as f:
        a = json.load(f)
        for c_id, line in enumerate(a):
            for item in line:
                sentences.append(item['utterance'])
                emotion_labels.append(item['emotion'])
                conv_id.append(split[:2] + '_c' + str(c_id))
                speakers.append(item['speaker'])
            
            # u_id += 1
                
    data = pd.DataFrame(sentences, columns=['sentence'])
    data['sentence'] = data['sentence'].apply(lambda x: expand_text(x))
    data['sentence'] = data['sentence'].apply(lambda x: preprocess_text(x))

    data['emotion_label'] = emotion_labels
    data['speaker'] = speakers
    data['conv_id'] = conv_id

    
    return data

"add padding to the conversations"
def pad_collate(x, max_len):   
    
    x = tf.cast(x, tf.float32)
    x = x.to_tensor()
    pad_size = list(x.shape)
    pad_size[0] = max_len - x.shape[0]
    x = tf.concat([x, tf.zeros(pad_size)], 0)
        
    return x
"apply the padding in the datasets"
def apply_padding(data):
    # find longest sequence
    new_batch = []
        
    for batch in data:
        new_messages = []
        new_speakers = []
        new_y = []
        max_len = max([sublist.shape[0] for sublist in batch[0]])
        for index, res in enumerate(batch):
            for i in res:
                if index == 0:
                    new_messages.append(pad_collate(i, max_len))
                elif index == 1:
                    new_speakers.append(pad_collate(i, max_len))
                else:
                    new_y.append(pad_collate(i, max_len))
                    
        new_messages = np.array(new_messages)
        new_messages = tf.convert_to_tensor(new_messages)
        new_speakers = np.array(new_speakers)
        new_speakers = tf.convert_to_tensor(new_speakers)
        new_y = np.array(new_y)
        new_batch.append([new_messages, new_speakers, new_y])
    return new_batch
"build batch according to batch size and padding defined"
def build_batch(data, number_speakers, batch_size):
    
    dialogue = data[data["speaker"].apply(lambda x: len(set(x)) == number_speakers)]
    
    X = (dialogue['encoded_speaker'],dialogue['sequence'])
    y = dialogue['emotion_true'].values

    speakers=tf.ragged.constant(X[0])
    dataset_speakers = tf.data.Dataset.from_tensor_slices(speakers)

    messages=tf.ragged.constant(X[1])
    dataset_messages = tf.data.Dataset.from_tensor_slices(messages)

    y = tf.ragged.constant(y)
    dataset_y = tf.data.Dataset.from_tensor_slices(y)
    
    dataset = tf.data.Dataset.zip((dataset_messages, dataset_speakers, dataset_y))
    data = dataset.batch(batch_size, drop_remainder=True)
    new_data = apply_padding(data)

    return new_data
"tranform classes into 0 and 1 representations"
class MyLabelBinarizer:

    def __init__(self):
        self.lb = LabelBinarizer()

    def fit(self, X):
        # Convert X to array
        X = np.array(X)
        # Fit X using the LabelBinarizer object
        self.lb.fit(X)
        # Save the classes
        self.classes_ = self.lb.classes_

    def fit_transform(self, X):
        # Convert X to array
        X = np.array(X)
        # Fit + transform X using the LabelBinarizer object
        Xlb = self.lb.fit_transform(X)
        # Save the classes
        self.classes_ = self.lb.classes_
        if len(self.classes_) == 2:
            Xlb = np.hstack((Xlb, 1 - Xlb))
        return Xlb

    def transform(self, X):
        # Convert X to array
        X = np.array(X)
        # Transform X using the LabelBinarizer object
        Xlb = self.lb.transform(X)
        if len(self.classes_) == 2:
            Xlb = np.hstack((Xlb, 1 - Xlb))
        return Xlb

    def inverse_transform(self, Xlb):
        # Convert Xlb to array
        Xlb = np.array(Xlb)
        if len(self.classes_) == 2:
            X = self.lb.inverse_transform(Xlb[:, 0])
        else:
            X = self.lb.inverse_transform(Xlb)
        return X

"create embedding"
def load_pretrained_glove():
    print("Loading GloVe model, this can take some time...")
    glv_vector = {}
    f = open('glove.840B.300d.txt', encoding='utf-8')

    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float')
            glv_vector[word] = coefs
        except ValueError:
            continue
    f.close()
    print("Completed loading pretrained GloVe model.")
    return glv_vector
"apply the encoder"
def encode_labels(encoder, l):
    return encoder[l]
"create the test subset of data from loaded Friends data"
def create_test_dataset():
    test_data = create_sentences('Friends/friends_test.json', 'test')
    
    with open("emotion_label_encoder.pkl", "rb") as f:
        emotion_label_encoder = pickle.load(f)
    
    test_data['encoded_emotion_label'] = test_data['emotion_label'].map(lambda x: encode_labels(emotion_label_encoder, x))
    
    ## tokenize all sentences ##
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    test_sequence = tokenizer.texts_to_sequences(list(test_data['sentence']))

    test_data['sentence_length'] = [len(item) for item in test_sequence]
    
    max_num_tokens = 250

    test_sequence = pad_sequences(test_sequence, maxlen=max_num_tokens, padding='post')
    test_data['sequence'] = list(test_sequence)
    
    test_data['emotion_true'] = pd.get_dummies(test_data['encoded_emotion_label']).values.tolist()
    test_data['sequence'] = np.array(test_data['sequence'])
    
    #agregate by the data by conversation
    dialogue_test_data = test_data.groupby("conv_id").agg(list)
    #encode the speaker into 0 when there is no one talking and 1 when there is
    dialogue_test_data['encoded_speaker'] = dialogue_test_data['speaker'].apply(lambda s: MyLabelBinarizer().fit_transform(s))
    dialogue_test_data['sequence'] = dialogue_test_data['sequence'].apply(lambda s: np.array(np.array(s)))
    dialogue_test_data['encoded_emotion_label'] = dialogue_test_data['encoded_emotion_label'].apply(lambda s: np.array(np.array(s)))
    dialogue_test_data['encoded_speaker'] = dialogue_test_data['encoded_speaker'].apply(lambda s: np.array(np.array(s)))
    dialogue_test_data.reset_index(inplace=True)
    #save test data
    pickle.dump(dialogue_test_data, open('test_data.pkl', 'wb'))
    
    return dialogue_test_data

"create the train subset of data from loaded Friends data"
def create_train_dataset():
    
    train_data = create_sentences('Friends/friends_train.json', 'train')
    
    
    ## encode the emotion and dialog act labels ##
    all_emotion_labels =  set(train_data['emotion_label'])
    emotion_label_encoder, emotion_label_decoder = {}, {}


    for i, label in enumerate(all_emotion_labels):
        emotion_label_encoder[label] = i
        emotion_label_decoder[i] = label

    pickle.dump(emotion_label_encoder, open('emotion_label_encoder.pkl', 'wb'))
    pickle.dump(emotion_label_decoder, open('emotion_label_decoder.pkl', 'wb'))

    train_data['encoded_emotion_label'] = train_data['emotion_label'].map(lambda x: encode_labels(emotion_label_encoder, x))
    
    
    ## tokenize all sentences ##
    all_text = list(train_data['sentence'])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_text)
    pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))

    ## convert the sentences into sequences ##
    train_sequence = tokenizer.texts_to_sequences(list(train_data['sentence']))
    
    train_data['sentence_length'] = [len(item) for item in train_sequence]
    
    max_num_tokens = 250

    train_sequence = pad_sequences(train_sequence, maxlen=max_num_tokens, padding='post')


    train_data['sequence'] = list(train_sequence)
    #tranform the emotion label into 0 and 1 representations
    train_data['emotion_true'] = pd.get_dummies(train_data['encoded_emotion_label']).values.tolist()
    train_data['sequence'] = np.array(train_data['sequence'])
    
    #agregate by the data by conversation

    dialogue_train_data = train_data.groupby("conv_id").agg(list)
    dialogue_train_data['encoded_speaker'] = dialogue_train_data['speaker'].apply(lambda s: MyLabelBinarizer().fit_transform(s))
    dialogue_train_data['sequence'] = dialogue_train_data['sequence'].apply(lambda s: np.array(np.array(s)))
    dialogue_train_data['encoded_emotion_label'] = dialogue_train_data['encoded_emotion_label'].apply(lambda s: np.array(np.array(s)))
    dialogue_train_data['encoded_speaker'] = dialogue_train_data['encoded_speaker'].apply(lambda s: np.array(np.array(s)))

    dialogue_train_data.reset_index(inplace=True)

    #save train data
    pickle.dump(dialogue_train_data, open('train_data.pkl', 'wb'))

    if not os.path.exists('glv_embedding_matrix'):
        ## save pretrained embedding matrix ##
        glv_vector = load_pretrained_glove()
        word_vector_length = len(glv_vector['the'])
        word_index = tokenizer.word_index
        inv_word_index = {v: k for k, v in word_index.items()}
        num_unique_words = len(word_index)
        glv_embedding_matrix = np.zeros((num_unique_words+1, word_vector_length))
    
        for j in range(1, num_unique_words+1):
            try:
                glv_embedding_matrix[j] = glv_vector[inv_word_index[j]]
            except KeyError:
                glv_embedding_matrix[j] = np.random.randn(word_vector_length)/200
    
        np.ndarray.dump(glv_embedding_matrix, open('glv_embedding_matrix', 'wb'))
    print ('Done. Completed preprocessing.')

    return dialogue_train_data
