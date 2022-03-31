# -*- coding: utf-8 -*-
"""




"""
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as k
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from utils import build_batch, convert_tensor, create_train_dataset



class AttentionBlock(k.layers.Layer):
    def __init__(self, D_g, **kwargs):
        super(AttentionBlock, self).__init__()
        self.D_g = D_g
        self.dense = k.layers.Dense(self.D_g)

    def call(self, tr, hg_all):
        hg = tf.convert_to_tensor(hg_all, dtype="float32")  # Hg = (32, n_iterations, 150)
        hg = tf.transpose(hg, [0, 2, 1]) #hg = (150, n_iterations, 32)
 
        tr = self.dense(tr)  # (32, 1, 150)
        tr = tf.expand_dims(tr, 1)
    

        score = tf.matmul(tr, hg, transpose_b=False)
        a_t = tf.nn.softmax(score, axis=0)  # 32, 1, 2

        aux = tf.transpose(hg, [0, 2, 1])  # 1 , 2, 150
        context = tf.matmul(a_t, aux)  # 150, 1, 32

        return context[:, 0, :]  # 32, 150

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "D_g": self.D_g
        })
        return config
    


def init_parameters(speakers, batch_size, D_p, D_g, D_e):
    number_of_speakers = speakers.shape[1]

    list_hp_parties = [np.zeros(
        (batch_size, D_p)) for x in range(number_of_speakers)]  # pur chaque party array (batch, hidden)
 

    hg = tf.zeros((batch_size, D_g))

    list_hg = [hg]

    he = tf.zeros((batch_size,D_e))

    return number_of_speakers, list_hp_parties, hg, list_hg, he


def select_party(hp_parties, who_talk):
    hp_prev = [] # np.zeros((batch_size, D_p))


    for e, (state, id) in enumerate(zip(hp_parties, who_talk)):
        hp_prev.append(state[tf.get_static_value(id)]) # récupère les états caché pour les batch previous
    hp_prev = tf.convert_to_tensor(hp_prev, dtype="float32")
    return hp_prev


def create_model(D_g, D_p, D_e, D_c, n_classes, glv_embedding_matrix, cnn_output_size, max_num_tokens, filters, kernel_sizes,dropout,batch_size ):

    vocab_size, embedding_dim = glv_embedding_matrix.shape
    # CREATE LAYER

    attention = AttentionBlock(D_g)

    embedding = k.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                   input_length=max_num_tokens, weights=[glv_embedding_matrix])
    convs1 = k.layers.Conv1D(filters,
                             kernel_sizes[0],
                             activation='relu')
    convs2 = k.layers.Conv1D(filters,
                             kernel_sizes[1],
                             activation='relu')
    convs3 = k.layers.Conv1D(filters,
                             kernel_sizes[2],
                             activation='relu')

    pooling = k.layers.GlobalMaxPooling1D()
    concatanate = k.layers.Concatenate()
    dropout = k.layers.Dropout(dropout)
    dense = k.layers.Dense(cnn_output_size, input_shape=(len(kernel_sizes) * filters,), activation='relu')


    inputs_tr = k.Input(shape=(1, max_num_tokens), batch_size=batch_size, dtype="float32", name='tr')
    hp_prev = k.Input(shape=(D_p), batch_size=batch_size, dtype="float32", name='hp_previous')
    hg = k.Input(shape=(D_g), batch_size=batch_size, dtype="float32", name='hg')
    hg_all = k.Input(shape=(None, D_g), batch_size=batch_size, dtype="float32", name='hg_all')
    he_prev = k.Input(shape=(D_e), batch_size=batch_size, dtype="float32", name='he_prev')

    # Execute embedding
    batch, num_utt, num_words = inputs_tr.shape

    x = tf.reshape(inputs_tr, [batch * num_utt, num_words])

    x = embedding(x)  # x size = (batch*num_utt, num_words = 250, embedding = 300)
    conv1_x = pooling(convs1(x))  # conv1_x size =  (num_utt * batch = 1, 50 )
    conv2_x = pooling(convs2(x))  # conv2_x size =  (num_utt * batch = 1, 50 )
    conv3_x = pooling(convs3(x))  # conv3_x size =  (num_utt * batch = 1, 50 )
    x = concatanate([conv1_x, conv2_x, conv3_x])  # x size =  (num_utt * batch = 1, 150 )
    x = dropout(x)
    tr = dense(x) # (32, 100)

    # concat uterance and
    c_tr_hp = tf.concat([hp_prev, tr], axis=-1)  # (32, 250)

    # Gru global

    hg_new = k.layers.GRUCell(D_g, dropout=0.5, name="GlobalGRU")(c_tr_hp, hg)  # (output, hidden state)
    hg_new = hg_new[0]  # get output

    # Compute attention
    context = attention(tr, hg_all)

    # party gru
    c_tr_ct = tf.concat([context, tr], axis=-1)

    hp_ = k.layers.GRUCell(D_p, dropout=0.5, name="PartyGRU")(c_tr_ct, hp_prev)
    hp_new = hp_[0]

    # emotion GRU
    he_ = k.layers.GRUCell(D_e, dropout=0.5, name="EmotionGRU")(hp_new, he_prev)
    he_new = he_[0]

    res = k.layers.Dense(2 * D_c, activation="relu")(he_new)

    y_pred_prob = k.layers.Dense(n_classes, activation="softmax", kernel_regularizer='l2')(res)

    model_gruglobal = k.Model(inputs=[inputs_tr, hg, hp_prev, hg_all, he_prev],
                                  outputs=[hg_new, hp_new, he_new, y_pred_prob], name="DialogueRNN")

    return model_gruglobal

def train_dialogue_RNN(data, num_epochs, batch_size ):

    D_g = 150  # taille hidden state gru global
    D_p = 150  # taille hidden state gru party
    D_e = 100  # taille hidden state gru emotion
    D_c = 100
    n_classes = 8
    glv_embedding_matrix = np.load(open('glv_embedding_matrix', 'rb') ,allow_pickle=True)
    cnn_output_size = 100
    max_num_tokens = 250
    filters = 50
    kernel_sizes = [3, 4, 5]
    dropout = 0.5

    optimizer = k.optimizers.Adam(learning_rate=1e-3, epsilon=1e-8)
    #loss_object = k.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    model = create_model(D_g, D_p, D_e, D_c, n_classes, glv_embedding_matrix, cnn_output_size, max_num_tokens, filters, kernel_sizes,dropout,batch_size)

    model.compile()

    epoch_loss = []
    epoch_accuracy = []

    for epoch in range(num_epochs):
        print("---------- Starting Training ---------- ")
        moyenne_batch_loss = []
        moyenne_accuracy = []

        # Training loop - using batches of 32
        for (batch, (input_sequence, qmask, label)) in enumerate(data):
            batch_accuracy = []
            input_sequence = convert_tensor(input_sequence)
            qmask = convert_tensor(qmask)
            label = tf.convert_to_tensor(label, dtype="float32")

            #input_sequence = tf.transpose(input_sequence, [1, 0, 2])
            #qmask = tf.transpose(qmask, [1, 0, 2])
            # print(qmask.shape)
            # print(label.shape)
            number_of_speakers, list_hp_parties, hg, list_hg, he  = init_parameters(qmask, batch_size, D_p, D_g, D_e)

            y_pred_all = []
            y_true_all = []
            y_pred_prob_all = []
            y_true_prob_all = []
            with tf.GradientTape(persistent=True) as tape:
                #for each id_msg in each batch 
                for id_msg in range(input_sequence.shape[1]):
                    #print("message")
                    #print(id_msg)
                    input_sequence_set = input_sequence[:, id_msg ,:]    
                    input_sequence_set = tf.transpose(tf.expand_dims(input_sequence_set, -1), [0, 2, 1]) #32, 1, 250
                    qmask_set = qmask[:, id_msg , :]
                    #print(label.shape)
                    label_set = label[:, id_msg, :]
         

                    #qmask = tf.reshape(qmask, [qmask.shape[1], qmask.shape[2]])

                    who_talk = tf.math.argmax(qmask_set, 1)
    
                    hp_parties = tf.convert_to_tensor(list_hp_parties, dtype="float32")
                  
    
                    hp_parties = tf.transpose(hp_parties, [1,0,2])
               
    
                    hp_prev = select_party(hp_parties, who_talk)   
                
                    tape.watch(input_sequence_set)
                    tape.watch(hp_prev)
                    hg_all = tf.convert_to_tensor(list_hg, dtype="float32")
                    
                    hg_all = tf.transpose(hg_all, [1, 0, 2])

                    tape.watch(hg_all)

                    #get last h_g from the list hg_all

                    hg = hg_all[: , -1 , :]

                    tape.watch(hg)  #
                    tape.watch(he)
                    tape.watch(model.trainable_variables)
                    hg_new, hp_new, he_new, y_prob = model([input_sequence_set, hg, hp_prev, hg_all, he])

                    list_hg.append(hg_new)
                    he = he_new
                    for i in range(number_of_speakers):
                        for batch in range(who_talk.shape[0]):
                            if who_talk[batch] == i:
                                list_hp_parties[i][batch,:] = hp_new[batch,:]

                    y_true_prob = label_set
    
                    y_pred = np.argmax(y_prob, axis=-1)
                    y_true = np.argmax(y_true_prob, axis=-1)
                    #print(y_true)
                    #print(y_pred)
                    
                    accuracy = accuracy_score(y_true, y_pred)
                    batch_accuracy.append(accuracy)

                    y_pred_prob_all.append(y_prob)
                    y_true_prob_all.append(y_true_prob)

                    y_pred_all.append(y_pred)
                    y_true_all.append(y_true)
                    
                    
                    #y_true = encode_label(label)
                result =  k.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)(y_true=y_true_prob_all, y_pred=y_pred_prob_all)
                loss =  tf.reduce_mean(result)
                
                    
            gradients = tape.gradient(loss, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            #print(gradients)
            #print(f"Step: {optimizer.iterations.numpy()},         Loss: {loss},       ")  # acc {acc}"
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            moyenne_batch_loss.append(tf.get_static_value(loss))
            moyenne_accuracy.append(np.mean(batch_accuracy))
            
        
        print(moyenne_accuracy)
        epoch_accuracy.append(np.mean(moyenne_accuracy))
        epoch_loss.append(np.mean(moyenne_batch_loss))
        model.save(f'saved_models/dialogue_rnn_model{epoch}.h5')
        plt.plot(epoch_loss, color="blue")
        plt.title(f"Epoch {epoch} Loss")
        plt.show()
        print('Epoch %d Loss %.3f Accuracy %.2f' % (epoch + 1, np.mean(moyenne_batch_loss), np.mean(moyenne_accuracy)))
        
    model.save('saved_models/dialogue_rnn_model.h5')
    plt.plot(epoch_loss, color="blue")
    plt.title("Epoch Loss")
    plt.show()

def predict_dialogue_RNN(data, batch_size):
    
    D_g = 150  # taille hidden state gru global
    D_p = 150  # taille hidden state gru party
    D_e = 100  # taille hidden state gru emotion
    
    y_predictions = []
    y_true_predictions = []
    moyenne_accuracy = []
    from tensorflow.keras.models import load_model

    model = load_model('saved_models/dialogue_rnn_model.h5', custom_objects={'AttentionBlock': AttentionBlock})
    

    
    for (batch, (input_sequence, qmask, label)) in enumerate(data):

        input_sequence = convert_tensor(input_sequence)
        qmask = convert_tensor(qmask)
        label = tf.convert_to_tensor(label, dtype="float32")
    
        number_of_speakers, list_hp_parties, hg, list_hg, he  = init_parameters(qmask, batch_size, D_p, D_g, D_e)
    
    
        y_pred_all = []
        y_true_all = []
        batch_accuracy = []
 
        for id_msg in range(input_sequence.shape[1]):
            input_sequence_set = input_sequence[:, id_msg ,:]    
    
            input_sequence_set = tf.transpose(tf.expand_dims(input_sequence_set, -1), [0, 2, 1]) #32, 1, 250
    
            qmask_set = qmask[:, id_msg , :]
            #print(label.shape)
            label_set = label[:, id_msg, :]
      
            who_talk = tf.math.argmax(qmask_set, 1)
    
            hp_parties = tf.convert_to_tensor(list_hp_parties, dtype="float32")
          
    
            hp_parties = tf.transpose(hp_parties, [1,0,2])
    
            hp_prev = select_party(hp_parties, who_talk)
          
            hg_all = tf.convert_to_tensor(list_hg, dtype="float32")
            
            hg_all = tf.transpose(hg_all, [1, 0, 2])
    
            hg = hg_all[: , -1 , :]
    
            hg_new, hp_new, he_new, y_prob = model([input_sequence_set, hg, hp_prev, hg_all, he])
    
            list_hg.append(hg_new)
            he = he_new
            for i in range(number_of_speakers):
                for batch in range(who_talk.shape[0]):
                    if who_talk[batch] == i:
                        list_hp_parties[i][batch,:] = hp_new[batch,:]
    
            y_true_prob = label_set
            
            y_pred = np.argmax(y_prob, axis=-1)
            y_true = np.argmax(y_true_prob, axis=-1)
            y_pred_new = []
            y_true_new = []
            for id in range(y_pred.shape[0]):
                if y_true[id] != 0:
                    y_pred_new.append(y_pred[id])
                    y_true_new.append(y_true[id])

            accuracy = accuracy_score(y_true_new, y_pred_new)
            batch_accuracy.append(accuracy)
            y_pred_all.append(y_pred_new)
            y_true_all.append(y_true_new)
            # y_true = encode_label(label)
        moyenne_accuracy.append(np.mean(batch_accuracy))
        y_predictions.append(y_pred_all)
        y_true_predictions.append(y_true_all)                                                                                  
    
    return y_predictions, y_true_predictions, np.mean(moyenne_accuracy)


if __name__ == "__main__":
    
  
    batch_size = 32 
    num_epochs = 55
    
    #dialogue_train_data, dialogue_test_data = create_dataset()
    if os.path.exists('train_data.pkl'):
        dialogue_train_data = np.load(open('train_data.pkl', 'rb') ,allow_pickle=True)
    else: 
        dialogue_train_data = create_train_dataset()

    #data1 = build_batch(dialogue_train_data, 1, 8)
    data2 = build_batch(dialogue_train_data, 2, batch_size)
    data3 = build_batch(dialogue_train_data, 3, batch_size)
    data4 = build_batch(dialogue_train_data, 4, batch_size)
    data = data2 + data3 + data4
  
    if not os.path.exists('saved_models/dialogue_rnn_model.h5'):
        train_dialogue_RNN(data, num_epochs, batch_size)
 

    
