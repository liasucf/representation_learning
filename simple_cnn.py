

import numpy as np
import os
import tensorflow as tf
from utils import build_batch, convert_tensor, create_train_dataset
import tensorflow.keras as k
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def create_model_CNN(n_classes, glv_embedding_matrix, cnn_output_size, max_num_tokens, filters, kernel_sizes, dropout,
                     batch_size):
    vocab_size, embedding_dim = glv_embedding_matrix.shape

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
    concatanate = k.layers.Concatenate()
    pooling = k.layers.GlobalMaxPooling1D()
    dropout = k.layers.Dropout(dropout)
    dense = k.layers.Dense(n_classes, activation="sigmoid", kernel_regularizer='l2')

    inputs_tr = k.Input(shape=(1, 250), batch_size=32, dtype="float32", name='tr')
    batch, num_utt, num_words = inputs_tr.shape
    x = tf.reshape(inputs_tr, [batch * num_utt, num_words])
    res_embed = embedding(x)
    res_conv1 = convs1(res_embed)
    res_conv2 = convs2(res_embed)
    res_conv3 = convs3(res_embed)

    res_pool1 = pooling(res_conv1)
    res_pool2 = pooling(res_conv2)
    res_pool3 = pooling(res_conv3)

    sequence = concatanate([res_pool1, res_pool2, res_pool3])  # x size =  (num_utt * batch = 1, 150 )
    # max_score = pooling(sequence)
    masked_document_embedding = dropout(sequence)
    print(masked_document_embedding.shape)
    output_probability = dense(masked_document_embedding)

    model_CNN = k.Model(inputs=inputs_tr, outputs=output_probability)

    return model_CNN

    return model_CNN


def train_CNN(data, num_epochs, batch_size):
    n_classes = 8
    glv_embedding_matrix = np.load(open('glv_embedding_matrix', 'rb'), allow_pickle=True)
    cnn_output_size = 100
    max_num_tokens = 250
    filters = 50
    kernel_sizes = [3, 4, 5]
    dropout = 0.5

    optimizer = k.optimizers.Adam(learning_rate=1e-3, epsilon=1e-8)
    # loss_object = k.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    model = create_model_CNN(n_classes, glv_embedding_matrix, cnn_output_size, max_num_tokens, filters, kernel_sizes,
                             dropout, batch_size)

    model.compile()
    print("model compile !")
    epoch_loss = []
    epoch_accuracy = []

    for epoch in range(num_epochs):
        print("---------- Starting Training ---------- ")
        moyenne_batch_loss = []
        moyenne_accuracy = []

        # Training loop - using batches of 32
        for (batch, (input_sequence, _, label)) in enumerate(data):
            batch_accuracy = []
            input_sequence = convert_tensor(input_sequence)
            label = tf.convert_to_tensor(label, dtype="float32")

            y_pred_all = []
            y_true_all = []
            y_pred_prob_all = []
            y_true_prob_all = []
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(model.trainable_variables)
                # for each id_msg in each batch
                for id_msg in range(input_sequence.shape[1]):
                    # print("message")
                    # print(id_msg)
                    input_sequence_set = input_sequence[:, id_msg, :]
                    input_sequence_set = tf.transpose(tf.expand_dims(input_sequence_set, -1), [0, 2, 1])  # 32, 1, 250
                    label_set = label[:, id_msg, :]
                    y_prob = model(input_sequence_set)

                    y_true_prob = label_set

                    y_pred = np.argmax(y_prob, axis=-1)
                    y_true = np.argmax(y_true_prob, axis=-1)

                    accuracy = accuracy_score(y_true, y_pred)
                    batch_accuracy.append(accuracy)

                    y_pred_prob_all.append(y_prob)
                    y_true_prob_all.append(y_true_prob)

                    y_pred_all.append(y_pred)
                    y_true_all.append(y_true)
                    # y_true = encode_label(label)
                result = k.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)(
                    y_true=y_true_prob_all, y_pred=y_pred_prob_all)
                loss = tf.reduce_mean(result)
            

            gradients = tape.gradient(loss, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            # print(gradients)
            # print(f"Step: {optimizer.iterations.numpy()},         Loss: {loss},       ")  # acc {acc}"
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            moyenne_batch_loss.append(tf.get_static_value(loss))

            moyenne_accuracy.append(np.mean(batch_accuracy))

        epoch_accuracy.append(np.mean(moyenne_accuracy))
        epoch_loss.append(np.mean(moyenne_batch_loss))
        model.save(f'saved_models/cnn_model{epoch}.h5')
        
        print('Epoch %d Loss %.3f Accuracy %.2f' % (epoch + 1, np.mean(moyenne_batch_loss), np.mean(moyenne_accuracy)))

    model.save('saved_models/cnn_model.h5')

    plt.plot(epoch_loss, color="blue")
    plt.title("Epoch Loss")
    plt.show()

    return moyenne_batch_loss

def predict_CNN(data):
    model = tf.keras.models.load_model('saved_models/cnn_model.h5')

    y_predictions = []
    y_true_predictions = []
    moyenne_accuracy = []
    for (batch, (input_sequence, _, label)) in enumerate(data):
        input_sequence = convert_tensor(input_sequence)
        label = tf.convert_to_tensor(label, dtype="float32")

        y_pred_all = []
        y_true_all = []
        batch_accuracy = []

        # for each id_msg in each batch
        for id_msg in range(input_sequence.shape[1]):
            # print("message")
            # print(id_msg)
            input_sequence_set = input_sequence[:, id_msg, :]
            input_sequence_set = tf.transpose(tf.expand_dims(input_sequence_set, -1), [0, 2, 1])  # 32, 1, 250
            label_set = label[:, id_msg, :]

            y_prob = model(input_sequence_set)

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
    num_epochs = 100
    # dialogue_train_data, dialogue_test_data = create_dataset()
    if os.path.exists('train_data.pkl'):
        dialogue_train_data = np.load(open('train_data.pkl', 'rb') ,allow_pickle=True)
    else: 
        dialogue_train_data = create_train_dataset()

    # data1 = build_batch(dialogue_train_data, 1, 8)
    data2 = build_batch(dialogue_train_data, 2, batch_size)
    data3 = build_batch(dialogue_train_data, 3, batch_size)
    data4 = build_batch(dialogue_train_data, 4, batch_size)
    data_train = data2 + data3 + data4
    
    if not os.path.exists('saved_models/cnn_model.h5'):
        print("begin train")
        train_CNN(data_train, num_epochs, batch_size)